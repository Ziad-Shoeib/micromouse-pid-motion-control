"""
mouse_sim.py — shared simulation core for the Micromouse PID session (Part 2).

Everything the seven example notebooks need lives here:

    Config      all mouse / motor / encoder / maze numbers in one place
    Motor       brushed DC motor as a first-order lag (gain Km, time constant Tm)
                with dead-band, battery scaling and an optional load
    Encoder     integer quadrature counts (quantised) from wheel angle
    DiffDrive   two motors + two encoders + differential-drive kinematics
    Corridor    a straight run of maze cells with walls, posts and openings
    IRSensor    a reflective IR wall sensor: raw ~ A / (d^2 + c) + noise
    PID         discrete PID with output clamp, anti-windup and filtered D
    Profile     trapezoidal / triangular speed profile (moving setpoint)
    plotting    top-view, signal plots and GIF export helpers

Units: SI internally (m, m/s, rad).  Motor commands are PWM in [-255, 255]
because that is what students will write on their hardware.  Gains are
therefore in "PWM per count" (heading / forward loops) or "PWM per raw
sensor unit" (wall loop) so the numbers transfer to real firmware.

Every notebook starts with `from mouse_sim import *` and a `cfg = Config(...)`
cell.  Change the numbers in Config to match *your* mouse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

__all__ = [
    "Config", "Motor", "Encoder", "DiffDrive", "Corridor", "IRSensor",
    "PID", "Profile", "wall_contact_x", "plot_topview", "plot_signals",
    "animate_topview", "savefig", "FIG_DIR", "np", "plt", "math", "replace",
]

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class Config:
    # --- geometry ---------------------------------------------------------
    wheel_r: float = 0.016        # wheel radius [m]  (32 mm wheel)
    track: float = 0.080          # distance between wheel contact points [m]
    mouse_width: float = 0.100    # body width [m]  (rulebook max is 0.14)
    mouse_length: float = 0.100   # body length [m]
    # --- encoder ----------------------------------------------------------
    cpr: int = 12                 # encoder counts per motor-shaft revolution
    gear: float = 30.0            # gearbox ratio motor:wheel
    # --- motor (first-order lag) -----------------------------------------
    km: float = 1.2               # wheel rim speed at PWM 255, full battery [m/s]
    tm: float = 0.08              # motor time constant [s]
    deadband: int = 12            # PWM below which the motor does not move
    v_nom: float = 8.4            # battery voltage the gains were tuned at [V]
    v_batt: float = 8.4           # actual battery voltage [V]
    mismatch: float = 0.0         # right motor gain = (1 - mismatch) * left
    # --- maze (VICTORIS 4.0 / classic IEEE) --------------------------------
    cell: float = 0.180           # cell pitch [m]
    wall_t: float = 0.012         # wall / post thickness [m]
    # --- simulation -------------------------------------------------------
    dt: float = 0.001             # control / integration step [s]  (1 kHz)
    pwm_max: int = 255

    # derived ------------------------------------------------------------
    @property
    def counts_per_rev(self) -> float:
        return self.cpr * self.gear

    @property
    def counts_per_m(self) -> float:
        return self.counts_per_rev / (2 * math.pi * self.wheel_r)

    @property
    def counts_per_mm(self) -> float:
        return self.counts_per_m / 1000.0

    @property
    def counts_per_cell(self) -> float:
        return self.counts_per_m * self.cell

    @property
    def counts_per_deg(self) -> float:
        """Encoder DIFFERENCE (L - R) per degree of in-place turn.
        Each wheel travels (track/2)*theta in opposite directions, so L - R = track*theta."""
        return self.counts_per_m * (math.pi * self.track / 180.0)

    @property
    def wheel_counts_per_deg(self) -> float:
        """Counts travelled by ONE wheel per degree of in-place turn (half of counts_per_deg)."""
        return self.counts_per_deg / 2.0

    @property
    def passage(self) -> float:
        return self.cell - self.wall_t          # 0.168 m

    @property
    def clearance(self) -> float:
        return (self.passage - self.mouse_width) / 2.0

    def summary(self) -> str:
        return (
            f"counts/rev = {self.counts_per_rev:.0f}   counts/mm = {self.counts_per_mm:.2f}   "
            f"counts/cell = {self.counts_per_cell:.0f}   counts/deg (L-R) = {self.counts_per_deg:.2f}\n"
            f"passage = {self.passage*1000:.0f} mm   clearance per side = {self.clearance*1000:.0f} mm   "
            f"battery = {self.v_batt:.1f} V / {self.v_nom:.1f} V"
        )


# --------------------------------------------------------------------------
# Motor, encoder, robot
# --------------------------------------------------------------------------

class Motor:
    """First-order lag: omega -> km_rad * u_eff with time constant tm.

    u is the PWM command (-255..255). Dead-band removes small commands.
    load is an equivalent speed loss [m/s at the rim] (e.g. an incline or
    a finger on the wheel) that pushes the steady-state speed down.
    """

    def __init__(self, cfg: Config, gain_scale: float = 1.0):
        self.cfg = cfg
        self.gain_scale = gain_scale
        self.omega = 0.0            # wheel angular speed [rad/s]
        self.load = 0.0             # rim speed loss [m/s]
        self.demand = 0.0           # rim-speed error driving the motor (proportional to torque)

    def step(self, pwm: float) -> float:
        cfg = self.cfg
        pwm = max(-cfg.pwm_max, min(cfg.pwm_max, pwm))
        u = 0.0 if abs(pwm) < cfg.deadband else pwm / cfg.pwm_max
        batt = cfg.v_batt / cfg.v_nom
        target_rim = cfg.km * self.gain_scale * batt * u - self.load
        target = target_rim / cfg.wheel_r
        self.demand = target_rim - self.omega * cfg.wheel_r   # ~ torque being applied [m/s of rim-speed error]
        self.omega += cfg.dt / cfg.tm * (target - self.omega)
        return self.omega


class Encoder:
    """Quantised counts from accumulated wheel angle."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.angle = 0.0            # accumulated wheel angle [rad]

    def update(self, omega: float) -> int:
        self.angle += omega * self.cfg.dt
        return self.counts

    @property
    def counts(self) -> int:
        return int(math.floor(self.angle / (2 * math.pi) * self.cfg.counts_per_rev))


class DiffDrive:
    """Two motors, two encoders, unicycle kinematics.

    slip: fraction of wheel motion NOT transferred to the floor (0..1),
          sampled uniformly in [0, 2*slip] each step so the mean is `slip`.
    """

    def __init__(self, cfg: Config, slip: float = 0.0, seed: int | None = None,
                 traction_dv: float | None = None, slip_max: float = 0.30):
        """traction_dv: crude traction limit.  Motor torque is proportional to the rim-speed error inside the
        motor (Motor.demand).  When |demand| exceeds traction_dv [m/s] the tyre slips, linearly up to slip_max
        at 2*traction_dv.  None = no torque-dependent slip.  (0.5 m/s ~ 6 m/s^2 of demanded acceleration
        with Tm = 0.08 s.)"""
        self.cfg = cfg
        self.traction_dv = traction_dv
        self.slip_max = slip_max
        self.mL = Motor(cfg, 1.0)
        self.mR = Motor(cfg, 1.0 - cfg.mismatch)
        self.eL = Encoder(cfg)
        self.eR = Encoder(cfg)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.slip = slip
        self.rng = np.random.default_rng(seed)
        self.v = 0.0
        self.w = 0.0

    @property
    def counts(self) -> tuple[int, int]:
        return self.eL.counts, self.eR.counts

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta

    def step(self, pwmL: float, pwmR: float) -> None:
        cfg = self.cfg
        wL = self.mL.step(pwmL)
        wR = self.mR.step(pwmR)
        self.eL.update(wL)
        self.eR.update(wR)
        if self.slip > 0:
            sL = 1.0 - self.rng.uniform(0, 2 * self.slip)
            sR = 1.0 - self.rng.uniform(0, 2 * self.slip)
        else:
            sL = sR = 1.0
        if self.traction_dv is not None:                        # torque-dependent (traction-limited) slip
            for side, mot in (("L", self.mL), ("R", self.mR)):
                over = (abs(mot.demand) - self.traction_dv) / self.traction_dv
                if over > 0:
                    f = 1.0 - self.slip_max * min(1.0, over)
                    if side == "L": sL *= f
                    else:           sR *= f
        vL = wL * cfg.wheel_r * sL
        vR = wR * cfg.wheel_r * sR
        self.v = 0.5 * (vL + vR)
        self.w = (vR - vL) / cfg.track
        self.x += self.v * math.cos(self.theta) * cfg.dt
        self.y += self.v * math.sin(self.theta) * cfg.dt
        self.theta += self.w * cfg.dt


# --------------------------------------------------------------------------
# Maze corridor and IR sensor
# --------------------------------------------------------------------------

class Corridor:
    """A straight corridor of `cells` cells along +x, centred on y = 0.

    Walls run along x at y = +/- passage/2 (inner faces).  Posts sit at every
    cell boundary x = k*cell, flush with the walls (as in the real maze).
    `missing` is a list of (cell_index, 'L'|'R') openings (cells numbered 0..).
    An optional end wall closes the corridor at x = cells*cell.
    """

    def __init__(self, cfg: Config, cells: int = 6, missing=(), end_wall: bool = False):
        self.cfg = cfg
        self.cells = cells
        self.missing = set(missing)
        self.segments: list[tuple[float, float, float, float]] = []   # x1,y1,x2,y2
        self.posts: list[tuple[float, float]] = []                    # centre x,y
        half = cfg.passage / 2
        t = cfg.wall_t
        for k in range(cells):
            x0, x1 = k * cfg.cell + t / 2, (k + 1) * cfg.cell - t / 2
            if (k, "L") not in self.missing:
                self.segments.append((x0, +half, x1, +half))
            if (k, "R") not in self.missing:
                self.segments.append((x0, -half, x1, -half))
        for k in range(cells + 1):
            for side in (+1, -1):
                cx, cy = k * cfg.cell, side * (half + t / 2)
                self.posts.append((cx, cy))
                # post = square of side t, add its 4 faces as segments
                self.segments += [
                    (cx - t/2, cy - t/2, cx + t/2, cy - t/2),
                    (cx + t/2, cy - t/2, cx + t/2, cy + t/2),
                    (cx + t/2, cy + t/2, cx - t/2, cy + t/2),
                    (cx - t/2, cy + t/2, cx - t/2, cy - t/2),
                ]
        if end_wall:
            xe = cells * cfg.cell - t / 2
            self.segments.append((xe, -half, xe, +half))
        self._seg = np.array(self.segments, dtype=float)

    def ray_distance(self, x: float, y: float, angle: float, max_range: float = 0.30) -> float:
        """Distance from (x, y) along `angle` to the nearest wall/post face."""
        dx, dy = math.cos(angle), math.sin(angle)
        x1, y1, x2, y2 = self._seg[:, 0], self._seg[:, 1], self._seg[:, 2], self._seg[:, 3]
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x1 - x) * ey - (y1 - y) * ex) / denom     # along the ray
            s = ((x1 - x) * dy - (y1 - y) * dx) / denom     # along the segment
        ok = (np.abs(denom) > 1e-12) & (t > 0) & (s >= 0) & (s <= 1)
        if not ok.any():
            return max_range
        return float(min(t[ok].min(), max_range))

    def draw(self, ax):
        cfg = self.cfg
        for (x1, y1, x2, y2) in self.segments:
            ax.plot([x1 * 1000, x2 * 1000], [y1 * 1000, y2 * 1000], color="0.25", lw=3)
        for (cx, cy) in self.posts:
            t = cfg.wall_t * 1000
            ax.add_patch(Rectangle((cx * 1000 - t/2, cy * 1000 - t/2), t, t, color="firebrick"))
        for k in range(1, self.cells):
            ax.axvline(k * cfg.cell * 1000, color="0.8", lw=0.8, ls=":")


class IRSensor:
    """Reflective IR sensor.  raw = A / (d^2 + c) + N(0, sigma), clipped to 0..1023.

    mount: (dx, dy) of the sensor in the body frame [m]; angle: relative to heading.
    With the defaults (45 deg sensors mounted 40 mm off-centre), a wall at the
    nominal 34 mm body clearance reads ~500 and 'no wall' (ray hits nothing
    within 0.30 m) reads ~25.
    """

    def __init__(self, mount=(0.04, 0.04), angle=math.radians(45),
                 A=2.2, c=0.0005, sigma=4.0, seed: int | None = None):
        self.mount = mount
        self.angle = angle
        self.A, self.c, self.sigma = A, c, sigma
        self.rng = np.random.default_rng(seed)

    def distance(self, corridor: Corridor, pose) -> float:
        x, y, th = pose
        sx = x + self.mount[0] * math.cos(th) - self.mount[1] * math.sin(th)
        sy = y + self.mount[0] * math.sin(th) + self.mount[1] * math.cos(th)
        return corridor.ray_distance(sx, sy, th + self.angle)

    def raw_from_distance(self, d: float) -> float:
        raw = self.A / (d * d + self.c) + self.rng.normal(0, self.sigma)
        return float(min(1023.0, max(0.0, raw)))

    def read(self, corridor: Corridor, pose) -> float:
        return self.raw_from_distance(self.distance(corridor, pose))


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------

class PID:
    """Discrete PID with output clamp, integrator anti-windup and filtered D.

    update(e, dt):
        integ += e*dt                     unless the output is saturated AND
                                          e pushes further into saturation
        deriv  = low-pass filtered (e - e_prev)/dt   (time constant tf)
        u      = kp*e + ki*integ + kd*deriv, clamped to [umin, umax]
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, tf=0.0, umin=-255.0, umax=255.0,
                 d_on_measurement: bool = False):
        self.kp, self.ki, self.kd, self.tf = kp, ki, kd, tf
        self.umin, self.umax = umin, umax
        self.d_on_measurement = d_on_measurement
        self.reset()

    def reset(self):
        self.integ = 0.0
        self.e_prev = None
        self.y_prev = None
        self.dstate = 0.0
        self.saturated = False
        self.u = 0.0

    def update(self, e: float, dt: float, y: float | None = None) -> float:
        """e = setpoint - measurement.  Pass y (the measurement) when
        d_on_measurement=True so a setpoint jump does not produce a derivative kick."""
        # --- integral with anti-windup (clamping) ---
        if not (self.saturated and (e * self.u) > 0):
            self.integ += e * dt
        # --- derivative (filtered) ---
        if self.d_on_measurement and y is not None:
            raw_d = 0.0 if self.y_prev is None else -(y - self.y_prev) / dt
            self.y_prev = y
        else:
            raw_d = 0.0 if self.e_prev is None else (e - self.e_prev) / dt
        if self.tf > 0:
            a = (2 * self.tf - dt) / (2 * self.tf + dt)
            b = 2 * dt / (2 * self.tf + dt)
            self.dstate = a * self.dstate + b * raw_d
            deriv = self.dstate
        else:
            deriv = raw_d
        self.e_prev = e
        u = self.kp * e + self.ki * self.integ + self.kd * deriv
        self.saturated = u > self.umax or u < self.umin
        self.u = max(self.umin, min(self.umax, u))
        return self.u


class Profile:
    """Trapezoidal / triangular speed profile used as a moving setpoint.

    start(distance, v0, vmax, vend, accel)  -> resets x_set = 0, v_set = v0
    update(dt)                              -> advances v_set, x_set, done
    The braking decision uses REMAINING DISTANCE, never elapsed time.
    Works for any unit (m, mm, degrees, counts) as long as they are consistent.
    """

    def __init__(self):
        self.done = True
        self.x_set = 0.0
        self.v_set = 0.0

    def start(self, distance, v0, vmax, vend, accel):
        self.d = float(distance)
        self.vmax, self.vend, self.a = float(vmax), float(vend), float(accel)
        self.x_set = 0.0
        self.v_set = float(v0)
        self.done = self.d <= 0

    def update(self, dt: float):
        if self.done:
            return
        remaining = self.d - self.x_set
        brake_dist = (self.v_set ** 2 - self.vend ** 2) / (2 * self.a)
        if remaining <= brake_dist:
            self.v_set = max(self.v_set - self.a * dt, self.vend)
        elif self.v_set < self.vmax:
            self.v_set = min(self.v_set + self.a * dt, self.vmax)
        else:
            self.v_set = self.vmax
        self.x_set += self.v_set * dt
        if self.x_set >= self.d:
            self.x_set = self.d
            self.v_set = self.vend
            self.done = True

    def duration(self, dt: float) -> float:
        """Total time of the profile (runs a copy of itself)."""
        p = Profile()
        p.start(self.d, self.v_set, self.vmax, self.vend, self.a)
        t = 0.0
        while not p.done and t < 60:
            p.update(dt)
            t += dt
        return t


# --------------------------------------------------------------------------
# Helpers and plotting
# --------------------------------------------------------------------------

import os
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def savefig(fig, name: str, dpi: int = 130) -> str:
    """Save a figure into ./figures (created if needed) and return the path."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def wall_contact_x(xs, ys, cfg: Config):
    """First x [m] where the mouse body edge reaches the wall face, or None."""
    limit = cfg.passage / 2 - cfg.mouse_width / 2
    hit = np.where(np.abs(ys) >= limit)[0]
    return float(xs[hit[0]]) if len(hit) else None


def plot_topview(ax, cfg: Config, traces, corridor: Corridor | None = None,
                 cells: int = 6, title: str = "", show_body=True, legend_loc="upper left"):
    """traces: list of (label, xs, ys[, plot_kwargs]) in metres."""
    if corridor is None:
        corridor = Corridor(cfg, cells)
    corridor.draw(ax)
    half = cfg.mouse_width * 500
    for tr in traces:
        lab, xs, ys = tr[:3]
        kw = dict(lw=1.8)
        if len(tr) > 3:
            kw.update(tr[3])                     # optional matplotlib kwargs, e.g. {"ls": "--"}
        xs, ys = np.asarray(xs) * 1000, np.asarray(ys) * 1000
        line, = ax.plot(xs, ys, label=lab, **kw)
        if show_body:
            ax.fill_between(xs, ys - half, ys + half, color=line.get_color(), alpha=0.10)
    ax.set_aspect("equal")
    ax.set_xlim(-30, corridor.cells * cfg.cell * 1000 + 30)
    ax.set_ylim(-cfg.passage * 700, cfg.passage * 700)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if title:
        ax.set_title(title)
    if legend_loc:
        # legend below the axes so it never covers the walls
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8, ncol=2, frameon=False)


def plot_signals(t, signals: dict, title: str = "", figsize=(9, 5.5), sharex=True):
    """signals: {'ylabel': {label: array, ...}, ...}  -> stacked axes."""
    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    if n == 1:
        axes = [axes]
    for ax, (ylabel, series) in zip(axes, signals.items()):
        for lab, arr in series.items():
            ax.plot(t[: len(arr)], arr, label=lab, lw=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def animate_topview(cfg: Config, xs, ys, ths, filename: str, corridor: Corridor | None = None,
                    cells: int = 6, fps: int = 25, stride: int = 20, title: str = "",
                    extra_traces=()):
    """Write a GIF of the mouse body moving along (xs, ys, ths). stride = sim steps per frame."""
    if corridor is None:
        corridor = Corridor(cfg, cells)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    corridor.draw(ax)
    for lab, exs, eys in extra_traces:
        ax.plot(np.asarray(exs) * 1000, np.asarray(eys) * 1000, lw=1.4, ls="--", color="tab:red", alpha=0.8, label=lab)
    trail, = ax.plot([], [], lw=1.5, color="tab:blue")
    body = Rectangle((0, 0), cfg.mouse_length * 1000, cfg.mouse_width * 1000,
                     color="tab:blue", alpha=0.35)
    ax.add_patch(body)
    ax.set_aspect("equal")
    ax.set_xlim(-30, corridor.cells * cfg.cell * 1000 + 30)
    ax.set_ylim(-cfg.passage * 700, cfg.passage * 700)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    if title:
        ax.set_title(title)
    if extra_traces:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8, frameon=False)
    idx = list(range(0, len(xs), stride)) + [len(xs) - 1]

    def frame(i):
        k = idx[i]
        trail.set_data(np.asarray(xs[:k]) * 1000, np.asarray(ys[:k]) * 1000)
        x, y, th = xs[k] * 1000, ys[k] * 1000, ths[k]
        L, W = cfg.mouse_length * 1000, cfg.mouse_width * 1000
        # rectangle anchored at rear-left corner in body frame
        cx = x - (L / 2) * math.cos(th) + (W / 2) * math.sin(th)
        cy = y - (L / 2) * math.sin(th) - (W / 2) * math.cos(th)
        body.set_xy((cx, cy))
        body.angle = math.degrees(th)
        return trail, body

    anim = FuncAnimation(fig, frame, frames=len(idx), interval=1000 / fps, blit=True)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return filename

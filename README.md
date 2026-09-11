# PID and Motion Control for Micromouse

Simulation examples from the PID & Motion Control sessions delivered to Micromouse competitors at the **IEEE Mansoura Student Branch — VICTORIS Competition**.

Seven runnable Python notebooks that build a micromouse motion controller from scratch: from "why does my mouse drift into a wall?" to a tuned two-loop controller with wall-following, calibrated turns, and trapezoidal speed profiling. Everything runs in simulation — no hardware needed.


![Show_Motion](micromouse-pid-part2/ex04_gap_with_logic.gif)

---

## Sessions

| | Recording | Slides |
|---|---|---|
| **Session 1 — PID Theory** | [YouTube](https://youtu.be/fZIfx3rb_Go?si=xnfuJDpp-ZoFZULi) | [PDF](https://drive.google.com/file/d/1a2V8Cr_0x5RSZMZ4PaCNPFEyQ2AGm7aC/view?usp=sharing) |
| **Session 2 — Hands-On Motion Control** | [YouTube (Soon)](ADD_LINK) | [PDF](https://drive.google.com/file/d/1IeF5ag0EbwDxNP0zIbDa4qHunQOeI2Rm/view?usp=sharing) |

Session 1 covers the theory (open vs closed loop, what P, I and D each do, tuning methods, motors and encoders). Session 2 — this repository — puts it into practice on a simulated micromouse.

---

## Quick start

```bash
git clone https://github.com/Ziad_Shoeib/micromouse-pid-motion-control.git
cd micromouse-pid-motion-control
pip install -r requirements.txt
jupyter lab    
```

Open any `ex0N_*.ipynb` and **Run All**. The notebooks come pre-executed, so you can also just read them on GitHub without installing anything.

**Adapting it to your mouse:** every physical number (wheel size, gear ratio, encoder CPR, track width, battery voltage) lives in the `Config` class at the top of `mouse_sim.py`. Change them there and re-run any notebook to see how *your* mouse would behave.

---

## Examples

### `ex01_open_loop_drift` — Why you need feedback at all
Runs the mouse with no feedback: equal PWM to both motors.

**Learn:** two "identical" motors never are — an 8% gain mismatch puts the mouse into a wall within ~1.4 cells. And the same command travels a different distance as the battery drains. Open loop cannot work.

---

### `ex02_pid_sliders` — The PID law, one wheel
A single wheel position loop with a step target and a load disturbance.

**Learn:** what each term actually does — P alone overshoots and leaves an offset under load, D damps the
overshoot, I removes the offset (but too much I causes ringing). This is why the tuning order is
**P → D → I**.

---

### `ex03_straight_line` — The two-controller structure
The core of every micromouse: a **forward** loop on the encoder *average* (how far) and a **heading** loop
on the encoder *difference* (which way), combined by a mixer.

**Learn:** distance control and direction control are separate problems — a forward loop alone still
drives into the wall. Also: how to compute `counts_per_mm` and `counts_per_cell` for your own mouse.

---

### `ex04_wall_following` — Using the walls, safely
Simulated IR sensors in a corridor with a missing wall section and posts at every cell boundary.

**Learn:** encoders keep you straight relative to where you *started*; walls keep you straight relative to
the *maze*. The critical part is the fallback logic — without it, the controller confidently steers into
an opening, because "no wall" and "wall very far away" look identical to an IR sensor.

---

### `ex05_turning` — Calibrated 90° and 180° turns
A turn is the same heading loop with a non-zero target.

**Learn:** how to compute `counts_per_deg` from your wheel track, why wheel slip means the encoders lie
about the real angle, and the calibration procedure for finding an *effective* track width that makes
commanded turns match reality.

---

### `ex06_tuning_quiz` — How to actually tune it
The P → D → I procedure, plus a symptom quiz: six step responses, each with one thing wrong.

**Learn:** to diagnose from a plot — slow response, overshoot, oscillation, noisy derivative, integrator
windup, dead-band, and the effect of a draining battery. Each with the fix.

---

### `ex07_speed_profile` — Stopping where you meant to
Trapezoidal speed profiles used as a moving setpoint for the forward loop.

**Learn:** a step target asks for infinite acceleration — the tyres slip, the encoders over-count, and the
mouse stops short while reporting "arrived." A profile fixes it, and feed-forward makes it fast *and*
accurate. Also: above ~1 m/s, acceleration matters more than top speed.

---

## Taking it to hardware

The notebooks are deliberately language-neutral: each controller is a short block of logic that maps
directly onto firmware in C++ (Arduino / STM32 / ESP32 / RP2040) or MicroPython. See
[`hardware_migration_guide.md`](hardware_migration_guide.md) for the mapping table, the fixed-rate loop
skeleton, platform-specific notes, and a 12-step bring-up order with a pass test for each step.

---

## Repository contents

```
├── mouse_sim.py              # shared simulation core (motor, encoder, robot, sensors, PID, profile)
├── ex01 … ex07 (.ipynb)      # the seven examples, pre-executed
├── ex01 … ex07 (.py)         # same content as plain scripts (jupytext source)
├── figures/                  # all plots and animations produced by the notebooks
├── hardware_migration_guide.md
└── requirements.txt
```

**Requirements:** Python ≥ 3.10, numpy, matplotlib. `ipywidgets` is optional (sliders in ex02); everything
else works without it.

---

## Notes

- The maze dimensions used throughout (18 cm cells, 16.8 cm passages, 14 cm mouse limit) follow the VICTORIS Micromouse rulebook. Check the current year's rulebook for your competition.
- The simulation is intentionally simplified — a first-order motor model, a crude slip model, and an idealised IR sensor. It is built to teach controller behaviour, not to predict your mouse's exact numbers.

---

*Questions or corrections are welcome — open an issue.*

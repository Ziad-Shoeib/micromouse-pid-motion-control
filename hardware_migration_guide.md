# From the notebooks to your mouse
## A migration guide for C++ (Arduino / STM32 / ESP32 / RP2040) and MicroPython — Block 8 handout

The notebooks were written so that every controller has a one-to-one twin in firmware. This guide is the map.
It is deliberately language-neutral: the *logic* is the deliverable; the syntax is yours.

---

## 1. What maps to what

| In the notebooks (`mouse_sim.py`) | On the mouse | Notes |
|---|---|---|
| `Config` (CPR, gear, wheel_r, track, cell) | `config.h` / `config.py` — one file of constants | `counts_per_mm`, `counts_per_cell`, `counts_per_deg` computed once at start-up; print them on boot. |
| `DiffDrive.step(pwmL, pwmR)` | the real world | Nothing to write — this is the part you *replace* with hardware. |
| `Encoder.counts` | quadrature counters (interrupts / hardware timer / PIO) | Copy both counts atomically at the top of every loop. |
| `IRSensor.read()` | ADC read of the phototransistor (pulsed emitter) | Return raw 0–1023/4095; keep the three-case logic of EX4 unchanged. |
| `PID.update(e, dt)` | `pid.h` / `pid.py` — copy the 12 lines | Same clamp, same anti-windup, same filtered D. |
| `Profile.update(dt)` | `profile.h` / `profile.py` — copy the 10 lines | Same class for distance (mm) and angle (deg). |
| `run_straight()` loop body (EX3) | the 1 kHz control ISR / timer callback | Read → two PIDs → mixer → PWM. Nothing else in it. |
| `wall_error()` (EX4) | same function, inside the loop | Thresholds `TH`, `L_NOM`, `R_NOM` from a calibration routine, stored in flash/EEPROM. |
| `turn_in_place()` (EX5) | a "motion primitive" the maze solver calls | Together with `move_cells(n)` this is the whole interface to the solver. |
| `plt.plot(...)` | serial/BLE CSV telemetry at 50–100 Hz | Without this you cannot do EX6 on hardware. |

---

## 2. The loop skeleton — the only structural rule that matters

**A fixed-rate control loop, driven by a hardware timer. The main loop only logs and decides.**

```
setup():
    init encoders, motors (PWM ≥ 20 kHz), ADC, timer at CONTROL_HZ (1000)
    load calibration constants; compute counts_per_mm, counts_per_deg
    pid_fwd, pid_head, pid_wall = PID(...); profile = Profile()

timer_isr()  every 1/CONTROL_HZ:           # <- EX3/EX4/EX5/EX7 live here, and nowhere else
    cL, cR   = read_encoders_atomic()
    L, R     = read_wall_sensors()          # or every Nth tick if ADC time is limited
    profile.update(dt)
    target   = profile.x_set * counts_per_mm
    dist     = (cL + cR) / 2 ;  diff = cL - cR
    fwd      = pid_fwd.update(target - dist, dt)
    turn     = steering(diff, L, R)          # EX4 three-case logic, or heading hold, or turn target (EX5)
    pwm_scale = V_NOM / read_battery()       # EX6 battery compensation
    set_motors(clamp((fwd + turn) * pwm_scale), clamp((fwd - turn) * pwm_scale))
    telemetry_push(t, dist, diff, L, R, fwd, turn)

main loop():
    telemetry_flush()                        # print CSV lines, parse "p 2.0 / step / move 3 / turn 90"
    maze solver: when profile.done -> decide next move -> profile.start(...)
```

**Rules that follow from it**
1. Never `delay()`/`sleep()` inside the ISR. Never allocate memory, print, or use floating-point-heavy library calls there.
2. `dt` is the *actual* period. If the platform cannot hold 1 kHz, run 500 or 250 Hz — but measure it, and retune (Ki, Kd scale with dt).
3. Reset all PID integrators and `e_prev` on every mode change (move → turn → move).
4. Cap the steering term (±120 PWM in EX3, ±30 % for the wall term in EX4) — steering must never take the motors from the forward loop.

---

## 3. Platform notes (choose your column)

| | **C++ (Arduino framework, STM32/ESP32/RP2040/AVR)** | **MicroPython (RP2040 Pico, ESP32)** |
|---|---|---|
| Fixed-rate loop | Hardware timer ISR: `IntervalTimer` (Teensy), `hw_timer` (ESP32), `HardwareTimer` (STM32), `TimerOne`/`TCA` (AVR) | `machine.Timer(period=…, callback=…)`. Realistic: **200–500 Hz**. Measure `dt` with `time.ticks_us()` inside the callback; do not assume the period. |
| Encoders | Pin-change interrupts on A and B (X4) or A only (X1, EX3 slide). Use `volatile` counts; copy with interrupts disabled. | Pico: use **PIO quadrature** (a ready `rp2` PIO program) — Python interrupts are too slow above a few kHz of edges. ESP32: `PCNT` peripheral via `machine.Counter`/`ESP32` module. |
| PWM | 20–25 kHz to be inaudible; 8–10 bit resolution is enough. `analogWrite` on AVR is 490/980 Hz — change the timer prescaler or accept the whine. | `machine.PWM(pin, freq=20000)`, `duty_u16()`. |
| H-bridge / direction | Sign → IN1/IN2 (or DIR pin), magnitude → PWM. Brake vs coast when PWM = 0: prefer brake for position holding. | Same. |
| IR sensors | Pulse the emitter: read ADC with emitter off, then on; use the difference (kills ambient). Sequence 4 sensors ≈ 200–400 µs on a fast MCU — fits in a 1 kHz loop. | ADC reads are ~20–50 µs each in MicroPython; read sensors every 2nd–5th tick and keep the latest values (the notebooks' `sensor_hz` parameter is exactly this). |
| Floating point | Fine on Cortex-M4F/ESP32/RP2040 (RP2040 has no FPU but is fast enough at 1 kHz for two PIDs). On AVR keep maths in `float` but avoid `pow`, `sqrt` inside the ISR (precompute). | Floats are boxed objects; keep the ISR body short. Avoid creating lists/dicts in the callback. |
| Telemetry | `Serial.print` CSV from the *main loop* using a ring buffer filled by the ISR; 115200–921600 baud. | `print()` from the main loop from a small `array` ring buffer. |
| Parameter tuning without reflashing | Tiny serial command parser: `p 2.0`, `i 0`, `d 0.1`, `step`, `move 3`, `turn 90`, `cal` | Same; `sys.stdin.readline()` in the main loop. |
| Persistent calibration | EEPROM / flash sector: `counts_per_mm`, `W_eff`, `TH`, `L_NOM`, `R_NOM`, gains | `json` file on the internal filesystem. |
| RTOS (rulebook bonus) | FreeRTOS task at highest priority pinned to the timer tick gives you the deterministic loop; a second task does telemetry and the solver. Only worth it if you can *show* the loop jitter improved. | `_thread` is not real-time; stay with the timer callback. |

---

## 4. Migration order (do it in this sequence; each step has a pass/fail test)

| Step | Do | Pass test | Notebook twin |
|---|---|---|---|
| 1 | Encoders: count both wheels, print counts. Spin each wheel one full turn by hand. | Both read ±`counts_per_rev` (sign matches the forward direction). | EX3 arithmetic cell |
| 2 | Motors open loop: same PWM to both for 1 s on the floor. | Note which way it curves (weaker motor) and how far it goes at full vs low battery. | EX1 |
| 3 | Timer loop at target rate; toggle a pin in the ISR and measure with a scope/logic analyser, or count ticks per second over serial. | Rate within 1 %; main loop never starves. | — |
| 4 | Forward loop only: `move 1` with `Kp_f`, `Kd_f`; log a step test. Tune P → D. | Stops within ±5 counts, no ringing; still curves sideways. | EX3 scenario B, EX6 steps 1–2 |
| 5 | Heading loop, mouse on a stand (wheels free): command `diff = 0` and twist a wheel by hand. | Returns to 0 within ~100 ms, no buzz (filter D if it buzzes). | EX3 scenario D, EX6 quiz 3 |
| 6 | Both loops on the floor: `move 5`. | Straight within a few mm; add a small `Ki_head` if it drifts steadily. | EX3 scenario C |
| 7 | Turn: `turn 90` ×4, measure, set `W_eff`; check 2×90 vs 180. | ≤ 1° residual per turn, repeatable. | EX5 |
| 8 | Sensor calibration routine: centred → `L_NOM, R_NOM`; no wall → floor; `TH` halfway. Store. | Values stable across 10 reads; repeat at the venue. | EX4 part 1 |
| 9 | Wall-following in a straight corridor with both walls; then with one wall removed. | No wiggle; goes straight past a gap. | EX4 cases b, d |
| 10 | Profile: `move 3` with `v_max = 0.5`, `a = 2`. Compare stop error to the step. Then add feed-forward `Kff = 255 / v_at_255`. | Stop error < 5 mm; PWM never saturates at exploration speed. | EX7 |
| 11 | Battery scaling on; retest 6–7 at low charge. | Same behaviour at 8.4 V and 6.6 V. | EX6 battery figure |
| 12 | Safe-gains set on a button; E-stop cuts the H-bridge supply (not just software). | Rulebook compliance. | — |

Only after step 12: hand the motion primitives (`move_cells(n)`, `turn(±90)`, `turn(180)`) to the maze-solver code.

---

## 5. The three constants you will type wrong at least once

1. **`counts_per_deg` is on the difference `L − R`** → `counts_per_mm × π·W/180`. Per wheel it is half.
2. **Sign of `diff`**: with `diff = L − R`, a positive (left/CCW) turn needs `target_diff = −counts_per_deg × angle`.
3. **`dt`**: `Ki` and `Kd` in the notebooks assume `dt = 0.001 s`. At 250 Hz multiply `Ki` by 0.25 and `Kd` by 4 to
   keep the same behaviour — or simply retune, which you have to do anyway.

---

## 6. Interface to the maze-solver session

```
move_cells(n)          # profile forward n*180 mm, v_end = 0, wall-following active
turn(angle_deg)        # in-place, ±90 / 180, angular profile
front_wall_align()     # optional: drive gently to the front wall to re-zero distance
sensors.walls()        # -> (left, front, right) booleans from thresholds
```
The solver decides *what*; everything in this session executes *how*. Keep the two in separate files/modules —
the rulebook's "code quality and documentation" criterion is scored on exactly this separation.

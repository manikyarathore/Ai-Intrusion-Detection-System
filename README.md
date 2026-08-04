# UAV-IDS Two-Drone Extension — Implementation

A real, runnable implementation of Section 11 (see `UAV_IDS_Framework.md`): two Skydio X2
clones from MuJoCo Menagerie, sensor-instrumented, flying a main-drone mission while an
attacker drone executes one of 4 attacks, detected in real time by an RL-trained
PRHN-lite classifier.

## What's actually here vs. what's simplified

**Real:**
- The Skydio X2 MJCF is cloned directly from `google-deepmind/mujoco_menagerie` (not
  reconstructed from memory) and combined into a genuine two-body MuJoCo scene
  (`models/two_drone_scene.xml`) with the sensors added exactly where Section 11.2 specs
  them (GPS/GNSS, gyro, accelerometer, magnetometer, per-rotor actuator force).
- The flight controller's motor-mixing matrix is derived algebraically from the X2's
  actual rotor positions and yaw-torque gear coefficients in the MJCF — not hand-tuned.
- The RL loop is genuine REINFORCE (with a variance-reduction baseline), trained inside
  MuJoCo, with the reward/penalty structure specified in Section 11.5. Training curves
  and held-out per-attack recall (below) are from actual runs, not illustrative numbers.

**Simplified, on purpose, with reasons noted in-code:**
- **ArduPilot**: this does *not* compile/run the real ArduCopter firmware — that's a
  large C++ codebase with a heavy build (submodules, SITL toolchain) that isn't practical
  to stand up here. Instead, `sim/controller.py` implements the same architecture
  ArduCopter actually uses — position → velocity → attitude → rate → motor-mixer cascade
  — in numpy. `sim/env.py`'s docstrings explain exactly where a real JSON-FDM bridge to
  ArduCopter SITL would slot in if you wanted to swap this for the real firmware later.
- **Detector**: a linear-softmax head over 9 physics-residual features (Section 5.3a's
  minimum-viable data branch), not the full CNN/GRU+attention PRHN. This was a deliberate
  choice so the RL update (REINFORCE) stays a simple, auditable few lines of numpy. Swap
  point is `sim/detector.py::featurize()`.

## Layout

```
models/two_drone_scene.xml   the combined two-drone MJCF (verified to load & step in MuJoCo)
sim/controller.py            ArduCopter-style cascaded controller + motor mixer
sim/attacks.py                the 4 attacks, as direct corruptions of specific channels
sim/comm_link.py              software-layer latency/drop model for Attack 4
sim/attacker.py                attacker's station-keeping + attack-trigger policy
sim/detector.py                PRHN-lite: physics-residual features + softmax classifier
sim/env.py                     Gymnasium env wiring it all together + the RL reward
sim/calibrate.py               one-time feature-normalization calibration pass
sim/train_rl.py                supervised warm-start + RL fine-tuning + checkpoint selection
sim/run_scenario.py            runs one episode, logs JSON for the visualizer
runs/detector.npz              the trained detector weights (see results below)
runs/all_scenarios.json        5 logged episodes (normal + 4 attacks) for the visualizer
```

The **`two_drone_ids_visualizer.html`** file (top level) is a self-contained, paused-by-
default animation (Section 11.6) of those 5 logged episodes — open it directly in a
browser, pick a scenario, press Play.

## Running it yourself

```bash
pip install mujoco gymnasium numpy
python -m sim.train_rl --episodes 150 --seconds 14 --out runs/detector.npz
python -m sim.run_scenario --attack 1 --detector runs/detector.npz --out runs/episode.json
```

## Training results (held out, not the training seeds)

`sim/train_rl.py`'s pipeline is: (1) collect a calibration batch across normal + all 4
attacks, freeze feature normalization from it — an earlier version used an *online-
adapting* normalizer, which for a sustained attack gradually re-absorbed the elevated
residual as "the new normal" and silently erased its own detection sensitivity mid-attack;
(2) a class-balanced supervised warm-start on that same batch (pure REINFORCE from a
random init reliably starves the rarer attack classes of any gradient signal at all —
this fixes the cold start); (3) REINFORCE fine-tuning with a reward baseline; (4)
validation-based checkpoint selection, scoring each checkpoint by its *minimum* per-class
recall (not the mean — a checkpoint acing 3 classes and dropping a 4th to zero should
score worse than one that's moderately good at all 4), so a bad fine-tuning step can't
silently ship.

| | normal | gps_spoof | imu_corrupt | actuator_injection | comm_jam |
|---|---|---|---|---|---|
| **recall** | — | 88% | **23%** | 65% | 78% |
| **normal-phase false-positive rate** | 0.3–0.4% across all runs | | | |

**`imu_corrupt` is genuinely weak, and that's reported honestly rather than tuned away.**
Direct inspection of the feature distributions (gyro-jump, accel-deviation) shows real
overlap between this attack's signal and ordinary flight noise for a linear classifier —
this is a limitation of the deliberately-simple detector head, not a bug. The framework's
full PRHN (Section 5.3a's CNN/GRU + attention data branch) is expected to do meaningfully
better here; `featurize()` is the drop-in swap point if you want to try.

## Bugs found and fixed while building this (kept here for transparency)

1. The original physics-branch "prediction" differenced the reported GPS against itself,
   making the GPS-spoof residual mathematically zero regardless of the attack. Fixed with
   genuine accelerometer dead-reckoning as an independent estimate.
2. The controller's mass constant was hand-summed incorrectly (2.325 kg vs. the true
   simulated 1.325 kg, verified via `m.body_subtreemass`), causing ~1.75x thrust over-
   command and a visible altitude climb.
3. An earlier attacker pursuit policy (dynamic chase with velocity feedforward) was
   control-unstable and would fly off to 20+ meters altitude. Replaced with station-
   keeping at a precomputed intercept point along the main drone's known route.
4. Pure per-step REINFORCE from a cold start collapsed onto whichever classes got lucky
   early and never recovered on the others — fixed with the supervised warm-start +
   checkpoint-selection pipeline described above.

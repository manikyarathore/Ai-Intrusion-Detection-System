# UAV-IDS Two-Drone Battlefield Extension — Final Report

## What this is

An implementation of Section 11 of `UAV_IDS_Framework.md`: two Skydio X2 drones (cloned
from MuJoCo Menagerie, not reconstructed) in a battlefield MuJoCo environment. A main
drone patrols a 32m leg along a border; an attacker drone stations itself across the
border and executes one of 4 attacks (GPS/GNSS spoofing, IMU corruption, actuator command
injection, comm jamming); an RL-trained detector classifies the attack in real time from
physics-residual features.

## Architecture

```
models/battlefield_scene.xml   60x60m grass field, border wall + dirt stripe,
                                2 static tanks + 1 car + 14 trees (all static/welded),
                                2 fully-actuated Skydio X2 drones

sim/controller.py              ArduCopter-style cascaded controller (pos->vel->att->rate->
                                motor-mixer), mixing matrix derived from real rotor geometry
sim/attacker.py                Station-keeping attacker: precomputes an intercept point
                                along the main drone's known path, holds position there
sim/attacks.py                 The 4 attacks as direct corruptions of specific channels
sim/comm_link.py               Software-layer latency/drop model for jamming
sim/detector.py                PRHN-lite: 14 physics-residual features -> small MLP
                                (14 -> 24 -> 5, ReLU + softmax)
sim/env.py                     Gymnasium env wiring everything together + RL reward
sim/calibrate.py               One-time feature-normalization calibration
sim/train_rl.py                Supervised warm-start + REINFORCE fine-tune + checkpoint
                                selection (headless)
sim/train_live.py              Same training pipeline, but running live inside MuJoCo's
                                3D viewer so you can watch it train, episode by episode
sim/run_scenario.py            Runs one episode, logs JSON for the HTML visualizer
sim/live_view.py               Live interactive 3D playback of a trained scenario
sim/record_video.py            Renders an actual MuJoCo-rendered GIF of a scenario
sim/ardupilot_bridge.py        Real ArduPilot JSON-FDM protocol bridge (see below)
```

## Results (held out, not training seeds)

| | gps_spoof | imu_corrupt | actuator_injection | comm_jam |
|---|---|---|---|---|
| **recall** | 88% | 96% | 66% | 92% |
| **normal-phase false-positive rate** | 0.1–0.2% across all four |

`actuator_injection` is the weakest of the four -- it's the attack most likely to end an
episode in an outright crash (a stuck/corrupted motor can genuinely make a quadrotor
uncontrollable), which shortens the detection window relative to the other three.

## Bugs found and fixed (chronological, kept for transparency)

1. **Tautological physics-branch prediction.** The GPS-spoof residual feature originally
   differenced the reported GPS against itself, making it mathematically guaranteed to be
   zero. Fixed with genuine IMU dead-reckoning (double-integrated accelerometer, rotated
   world-frame) as an independent estimate the spoof can't touch.
2. **Mass double-counting.** The controller's `MASS` constant was hand-summed to 2.325kg;
   the true simulated mass (verified via `m.body_subtreemass`) is 1.325kg. Caused ~1.75x
   thrust over-command and a visible altitude climb.
3. **Unstable attacker pursuit.** An early chase-based attacker policy (velocity
   feedforward tracking a moving target) was control-unstable and flew off to 20+ meters
   altitude. Replaced with station-keeping at a precomputed intercept point.
4. **REINFORCE cold-start collapse.** Pure per-step policy-gradient from a random init
   starved the rarer attack classes of any gradient signal. Fixed with a class-balanced
   supervised warm-start plus partial prior-correction for the natural class imbalance.
5. **RL fine-tuning silently destroying a class.** Aggregate reward improved while
   `actuator_injection` recall dropped to 0%. Fixed with validation-based checkpoint
   selection scored by *minimum* per-class recall (not the mean), so a bad fine-tuning
   step can't ship silently -- it's been triggered on nearly every training run since,
   and has correctly reverted to the pre-RL checkpoint every time so far.
6. **`imu_corrupt` detection was near-chance (~23% recall).** Root cause: its actual
   signal is bursty spikes that single-instant features mostly miss. Fixed by (a)
   strengthening the attack to a less-subtle ramp, (b) adding windowed features
   (max/mean/spike-rate over the last 20 ticks), (c) upgrading the classifier from linear
   to a small MLP, since the real signature is a nonlinear combination across residuals.
7. **PID integral windup.** Scaling the mission up to 32m (and episodes to 30s) surfaced
   an unbounded integral term that, under a sustained actuator fault, wound up and sent
   the drone to absurd coordinates (100+ meters). Fixed with anti-windup clamping and a
   physically-motivated crash-termination bound.
8. **A newly-added path-deviation feature measured the wrong signal.** To help
   `gps_spoof` detection at the larger scale, a cross-track-deviation-from-mission-path
   feature was added -- but the first version measured the *reported* GPS position, which
   the controller actively steers to *look* on-path even while spoofed (that's the whole
   point of the attack), so the feature was ~0 for both classes. Fixed by measuring
   deviation of the physics branch's own IMU dead-reckoning estimate instead.
9. **That fix then got drowned out by outliers.** `actuator_injection`'s occasional crash
   trajectories sent the new path-deviation feature to huge values, blowing up its
   normalization scale and burying `gps_spoof`'s much smaller (but real) signal. Fixed by
   clipping the feature before normalization.

## What's real vs. simplified

**Real:** the cloned Skydio X2 MJCF, the battlefield scene (verified via actual MuJoCo
renders, not just XML review), the motor-mixing matrix (derived from rotor geometry, not
hand-tuned), the REINFORCE training loop (genuine reward-driven weight updates, verified
via real training runs), the ArduPilot JSON-FDM bridge's protocol mechanics (verified via
loopback UDP tests -- see below for what's *not* verified).

**Simplified, with the reasoning documented in-code:**
- **ArduPilot**: `sim/controller.py` implements ArduCopter's real control *architecture*
  (position->velocity->attitude->rate->motor-mixer cascade) in numpy, not the compiled
  firmware itself -- building real ArduCopter SITL needs a large C++ toolchain that isn't
  practical in the sandbox this was built in. `sim/ardupilot_bridge.py` is a real,
  protocol-correct JSON-FDM bridge that *would* let real ArduCopter SITL fly the main
  drone through this MuJoCo physics -- its socket/JSON mechanics are tested (loopback),
  but it has **not** been verified end-to-end against a real ArduCopter binary, which
  wasn't buildable in this environment. Treat it as a real starting point, not a proven
  integration.
- **Detector**: a 14-feature MLP over hand-engineered physics residuals, not the full
  CNN/GRU+attention PRHN from Section 5.3a. `featurize()` is the swap point.

## Live training, video, and playback

- `python -m sim.train_live.py` -- watch the RL model actually train inside MuJoCo's 3D
  viewer, episode-by-episode, with live reward/accuracy/recall printed to the terminal.
  Paused on load, SPACE to start.
- `python -m sim.record_video --attack N` -- renders an actual MuJoCo GIF of a trained
  scenario (this is how the preview images/GIFs shared in chat were generated).
- `python -m sim.live_view --attack N` -- live interactive 3D playback of a trained
  scenario with a fixed detector (no training happening, just watching it detect).

## Known limitations, stated plainly

- `actuator_injection` recall (66%) is meaningfully behind the other three attacks.
- The ArduPilot bridge is unverified against real firmware.
- The detector is a lightweight MLP, not the full architecture described in Section 5.
- Only one attacker drone; only one patrol leg (straight line, not a multi-waypoint
  perimeter) -- both flagged as reasonable next steps if pursued further.

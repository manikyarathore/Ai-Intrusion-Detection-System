# AI-Enhanced Intrusion Detection for UAV Systems — Implementation

A real, runnable MuJoCo simulation of a 2v2 UAV formation scenario: one drone is a
covert attacker embedded in one team; it attacks a drone on the opposing team; that
drone's own teammate detects the attack with a trained lightweight neural network,
runs a **Bayesian game decision** (Sedjelmaci et al., 2017) to decide whether to eject
it, and commits that decision to a **hash-chained blockchain ledger** (built following
the mechanisms surveyed in Jain et al., 2024) requiring a second drone's sign-off
before it takes effect. Everything — physics, attacks, detection, the Bayesian
decision, and training itself — runs through real MuJoCo simulation steps.

This document only describes what is actually implemented and tested in this
repository. Where something is simplified or unverified, that is stated explicitly.

---

## 1. The scenario

**Team A** (`a1` = attacker, `a2` = escort) and **Team B** (`b1` = victim, `b2` =
friend/monitor) fly a 2v2 formation over a 60m×60m battlefield split by a border wall
into two territories.

- `b1` flies a 32m patrol leg on the friendly side of the border (`x < 0`).
- `b2` flies wingman formation alongside `b1`, offset by `(3, 0, 0)` metres.
- `a1` is the embedded attacker: it stations itself at a precomputed intercept point
  along `b1`'s known route, across the border in enemy territory, and — once `b1`
  comes within 16m — executes one of 4 attacks against it.
- `a2` flies escort formation alongside `a1`, offset by `(0, 6, 0)` metres.
- `b2` does **not** get privileged information about `b1`. It runs its own trained
  detector against `b1`'s *reported* telemetry, received over a trusted inter-drone
  link (a simplifying assumption — this link itself is not modeled as its own
  attackable channel).

All four drones fly at a fixed 3.0m altitude with a real ArduCopter-style cascaded
controller (position → velocity → attitude → rate → motor mixer). Formation-holding
was measured directly, not just eyeballed: both wingmen hold their target offset to
within a few centimetres during flight.

## 2. The 4 attacks

Implemented in `sim/attacks.py` as direct corruptions of specific telemetry/actuator
channels — not abstract labels:

| Attack | What it corrupts |
|---|---|
| `gps_spoof` | `b1`'s reported GNSS position (ramping offset, up to 3m) |
| `imu_corrupt` | `b1`'s gyroscope + accelerometer readings (ramping bias + spikes) |
| `actuator_injection` | One of `b1`'s 4 motor commands (post-mixing, magnitude 4.0) |
| `comm_jam` | Latency/packet-loss injected into `b1`'s control loop (software-layer model) |

`a1` only triggers an attack once actually within 16 metres of `b1` and past its
scheduled onset time — this is computed from real inter-drone GPS positions each tick,
not scripted.

## 3. Detection: PRHN-lite

`sim/detector.py` — a small 2-layer MLP (14 → 24 → 5, ReLU + softmax) over 14
hand-engineered physics-residual features:

- Single-instant residuals: GPS-vs-dead-reckoning residual, gyro magnitude/jump, accel
  deviation from gravity, commanded-vs-applied actuator residual, comm latency/drop
  rate, magnetometer residual, speed.
- Windowed features (over the last 20 ticks): max gyro-jump, spike-rate, mean accel
  deviation, max actuator residual — added because `imu_corrupt`'s real signal is
  bursty spikes that single-instant features mostly miss.
- Path-deviation feature: cross-track distance of the physics branch's own IMU
  dead-reckoning estimate (not the possibly-spoofed GPS reading) from `b1`'s known
  mission path — added for `gps_spoof` detection at this map scale, clipped before
  normalization so `actuator_injection`'s occasional crash trajectories don't blow out
  the feature's scale.

The physics-branch position estimate (`gps_physics_pred`) is genuine accelerometer
dead-reckoning (double-integrated, rotated body→world by the real attitude quaternion)
— not a formula that differences the reported GPS against itself, which was an earlier
bug that made GPS-spoof detection mathematically impossible (see Section 8).

**Training pipeline** (`sim/train_rl.py` / `sim/train_live_four.py`):
1. Collect a calibration batch across normal flight + all 4 attacks; freeze feature
   normalization from it.
2. Class-balanced supervised warm-start (cross-entropy) on that batch, with a partial
   prior-correction back toward the natural (mostly-normal) class frequency.
3. REINFORCE fine-tuning with a reward baseline (reward/penalty structure below).
4. Validation-based checkpoint selection: score = **minimum** per-class recall (not the
   mean) minus a false-positive penalty, so a checkpoint can't hide a dead class behind
   good aggregate numbers. In every training run so far, the post-warm-start checkpoint
   has scored best — RL fine-tuning has not yet beaten it on this feature set, and
   checkpoint selection has correctly kept the warm-start checkpoint every time.

**Reward structure** (in both `env.py` and `env_four.py`): correct attack detection =
+3.0 plus an onset-speed bonus up to +2.0; correctly predicting "normal" during normal
flight = +0.5; a miss = −3.0; a false positive = −1.5.

## 3a. Measured results — and why raw recall is misleading here

Run it yourself:
```bash
python3 -c "
from sim.evaluate import evaluate_detailed, print_report
from sim.env_four import FourDroneIDSEnv
from sim.detector import PRHNLiteDetector
det = PRHNLiteDetector(seed=0); det.load('runs/detector_four.npz')
results, fpr = evaluate_detailed(det, lambda s: FourDroneIDSEnv(seed=s), seeds=range(3000,3012))
print_report(results, fpr)
"
```

**2v2 environment, 12 held-out seeds per attack:**

| attack | raw recall | post-onset recall | detection latency | detected in |
|---|---|---|---|---|
| gps_spoof | 0.897 | 0.991 | 2.17s | 12/12 runs |
| imu_corrupt | 0.950 | 0.981 | 0.06s | 12/12 runs |
| actuator_injection | 0.675 | **1.000** | 1.25s | 12/12 runs |
| comm_jam | 0.926 | 0.998 | 0.43s | 12/12 runs |

Normal-phase false-positive rate: **0.0000**.

**An earlier version of this README listed `actuator_injection` as "the weakest-detected
attack" at ~67% recall. That reading was wrong, and measuring latency directly is what
showed it.** Raw recall counts every attack-active tick equally, including the ticks
immediately after onset when a gradually-ramping attack is still physically
indistinguishable from normal flight — at 50 ticks post-onset, `actuator_injection`'s
accel-magnitude feature reads 0.06 against a normal-flight mean of 0.11, so there is
genuinely nothing to detect yet. Bucketing recall by time-since-onset shows it going
0% → 14% → 28% → **100%**, and staying at 100% for the rest of the attack.

So `actuator_injection` is not detected *worse*; it's detected **faster** than
`gps_spoof` (1.25s vs 2.17s). Its raw recall is lower only because it frequently
destabilizes the drone enough to end the episode early, making the unavoidable ramp-in
window a larger fraction of a shorter episode. `sim/evaluate.py` exists to report all
three numbers so this failure mode of single-metric reporting doesn't recur.



## 4. Bayesian ejection decision (`sim/bayesian_game.py`)

Implements the two-game structure from Sedjelmaci, Senouci & Ansari, *"Intrusion
Detection and Ejection Framework Against Lethal Attacks in UAV-Aided Networks: A
Bayesian Game-Theoretic Methodology"* (IEEE T-ITS, 2017):

- **Game 1 (IDS activation)**: should `b2` even activate monitoring against `b1`?
  Decided by comparing a computed Misbehavior Rate (MR) against a Bayesian-equilibrium
  threshold `B_attacker`, derived from the paper's profit equations (Eqs. 2–9) — the
  break-even point between the monitor's payoff from monitoring vs. staying idle.
- **Game 2 (IES ejection)**: should `b2` actually eject `b1`? Same construction over
  the paper's ejection-game profit equations (Eqs. 16–21), giving a second threshold
  `B_node`. Crucially, this only fires after MR has stayed over threshold for a
  **persistence window** (300 ticks / 3 seconds) — a drone that misbehaves briefly, or
  in oscillating bursts that never sustain, is **not** ejected. This is the paper's
  central distinction between *transitory* misbehavior (forgiven) and *permanent*
  misbehavior (ejected), and it's tested directly, not just implemented:
  - a 50-tick burst followed by 2000 normal ticks → not ejected
  - 5 repeated 50-tick bursts, oscillating, never sustained → not ejected
  - a genuinely sustained 300+ tick attack → ejected, at tick ~299

MR itself (`misbehavior_rate()`) is computed from `b2`'s own detector flags over a
sliding 100-tick window, using the paper's exponential MR1 form (Eq. 1).

**What's simplified**: the cost/profit parameters in `GameParams` (false-negative
rate, damage, monitoring/ejection cost, etc.) are set from this project's own measured
detector recall/FPR, not from an independently-sourced cost model — the paper doesn't
specify universal values for these either. The persistence-window mechanism is this
project's own way of operationalizing "permanent vs. transitory" from the paper's
qualitative description in Section II-B; the paper itself doesn't prescribe an exact
window length.

## 5. Blockchain ledger (`sim/blockchain_ledger.py`)

Built following the mechanisms surveyed in Jain, Barke, Garg et al., *"A Walkthrough
of Blockchain-Based Internet of Drones Architectures"* (IEEE IoT Journal, 2024).

**What's real:**
- Genuine SHA-256 block hash-chaining (each block's hash depends on its content *and*
  the previous block's hash).
- **Tamper-evidence that was actually tested, not just claimed**: mutating a committed
  block's data breaks `is_valid()`, including the harder case where the tamperer also
  recomputes that block's own hash afterward — still caught, because the *next*
  block's `prev_hash` link is now stale.
- A multi-signature-style commit gate, modeled on the paper's description of an n-m
  multisignature smart contract: an ejection block only commits once a quorum
  (default 2) of drones *other than the subject* have signed it. `b1` cannot sign a
  block accusing itself. In the live pipeline, `b2` (the accuser) and `a2` (an
  independent third party) sign; `a1` is deliberately never asked, since asking the
  actual attacker to confirm an accusation against its own target has an obvious
  conflict of interest.

**What's not implemented, stated plainly:** no real distributed consensus (PoW / PoS /
PBFT / DPoS) and no peer-to-peer networking — this is a single-process, in-memory
ledger shared by the drones in this simulation, not independently-running distributed
nodes. "Signatures" are HMAC-SHA256 keyed hashes per drone, not real asymmetric
cryptography (no ECDSA / public-key infrastructure). These are both explicitly future
scope, not something faked here.

## 6. Layout

```
models/four_drone_battlefield.xml   the 2v2 scene (default for the current architecture)
models/battlefield_scene.xml        the earlier 2-drone (1 attacker vs. 1 victim) battlefield
models/two_drone_scene.xml          the original small-scale (~8m mission) scene
models/assets/                      mesh + texture files the above need

sim/controller.py         ArduCopter-style cascaded controller + motor mixer
sim/attacks.py             the 4 attacks, as direct corruptions of specific channels
sim/comm_link.py            software-layer latency/drop model for comm_jam
sim/attacker.py              station-keeping + attack-trigger policy (used by a1)
sim/detector.py               PRHN-lite: physics-residual features + MLP classifier
sim/evaluate.py                detailed metrics: detection latency + post-onset recall (Section 3a)
sim/bayesian_game.py           Bayesian IDS/IES ejection decision (Section 4 above)
sim/blockchain_ledger.py        hash-chained ledger + multisig commit gate (Section 5)
sim/env_four.py                  FourDroneIDSEnv -- the 2v2 Gymnasium env (current default)
sim/env.py                        TwoDroneIDSEnv -- the earlier 1v1 Gymnasium env
sim/calibrate.py                   feature-normalization calibration pass (1v1 env)
sim/train_live_four.py              live 3D training for the 2v2 scenario (primary path)
sim/run_scenario_four.py             runs one 2v2 episode, logs JSON for the 2v2 HTML visualizer
sim/record_video_four.py              renders an actual MuJoCo GIF of a 2v2 scenario
sim/train_live.py                    live 3D training for the 1v1 scenario
sim/train_rl.py                       headless training for the 1v1 scenario + evaluate_recall()
sim/run_scenario.py                    runs one 1v1 episode, logs JSON for the HTML visualizer
sim/live_view.py                        live interactive 3D playback of a trained 1v1 scenario
sim/record_video.py                      renders an actual MuJoCo GIF of a 1v1 scenario
sim/ardupilot_bridge_four.py              real ArduPilot JSON-FDM bridge for the 2v2 scenario
sim/ardupilot_bridge.py                   real ArduPilot JSON-FDM protocol bridge (Section 8, 1v1)

runs/detector_four.npz    pretrained detector for the 2v2 scenario (supervised warm-start only)
runs/all_scenarios_four.json  5 logged 2v2 episodes (normal + 4 attacks) for the 2v2 HTML visualizer
runs/gps_spoof_demo_four.gif  an example MuJoCo-rendered recording of the 2v2 scenario
runs/detector.npz         pretrained detector for the 1v1 battlefield scenario
runs/detector_curves.npz  reward/accuracy training curves for the 1v1 detector
runs/all_scenarios.json   5 logged 1v1 episodes (normal + 4 attacks) for the 1v1 HTML visualizer
runs/gps_spoof_demo.gif   an example MuJoCo-rendered recording of the 1v1 scenario

four_drone_preview.png     an actual MuJoCo offscreen render of the 2v2 scene
battlefield_preview.png    an actual MuJoCo offscreen render of the 1v1 battlefield scene
four_drone_ids_visualizer.html  self-contained, paused-by-default 2D playback (2v2 scenario,
                                 shows MR/threshold bars, ejection streak, and the blockchain)
two_drone_ids_visualizer.html   self-contained, paused-by-default 2D playback (1v1 scenario)
REPORT.md                  project write-up covering the 1v1 architecture and its bug history
```

**Why both the 2v2 and 1v1 environments are still here:** the 2v2 scenario
(`env_four.py`, current default) is the architecture described in Sections 1–5 above.
The 1v1 scenario (`env.py`) is the earlier design it was built from. All tooling has
now been ported to 2v2 (`train_live_four.py`, `run_scenario_four.py`,
`record_video_four.py`, `four_drone_ids_visualizer.html`, `ardupilot_bridge_four.py`).

## 7. Running it

```bash
pip install mujoco gymnasium numpy
```

**Watch the 2v2 model train live, in 3D — the primary/recommended way to train:**
```bash
python -m sim.train_live_four --episodes 80 --out runs/detector_four.npz
```
Opens MuJoCo's native viewer running the actual training loop. Episode number, current
attack, reward, and step-accuracy print to the terminal live; every 10 episodes it
prints a full held-out per-class recall breakdown; ejection/blockchain events print the
moment they happen. **Paused on load — press SPACE to start.** Needs a display.

**Test the pretrained 2v2 detector without training anything:**
```bash
python3 -c "
from sim.env_four import FourDroneIDSEnv
from sim.detector import PRHNLiteDetector
from sim.attacks import ATTACK_NAMES

det = PRHNLiteDetector(seed=0)
det.load('runs/detector_four.npz')
for atk in [1,2,3,4]:
    env = FourDroneIDSEnv(seed=42)
    obs, info = env.reset(force_attack_id=atk, onset_frac=0.4, seed=42)
    term = trunc = False
    while not (term or trunc):
        action, p, cache = det.act(obs, explore=False)
        obs, r, term, trunc, info = env.step(action)
    fs = info['friend_status']
    print(ATTACK_NAMES[atk], 'ejected:', fs['ejected'], 'MR:', round(fs['mr'], 3),
          'blockchain valid:', env.blockchain.is_valid())
"
```

**Log a 2v2 scenario for the visualizer, or record a GIF of it:**
```bash
python -m sim.run_scenario_four --attack 1 --detector runs/detector_four.npz --out runs/episode_four.json
MUJOCO_GL=egl python -m sim.record_video_four --attack 1 --detector runs/detector_four.npz --out runs/demo_four.gif
```

**Open the 2v2 2D top-down visualizer**: open `four_drone_ids_visualizer.html` directly
in a browser — shows all 4 drones, the border, MR against both Bayesian thresholds, the
ejection persistence streak, and the blockchain ledger growing live. Paused on load.

**Train the older 1v1 scenario headless** (optional; `runs/detector.npz` already has a
pretrained model):
```bash
python -m sim.train_rl --episodes 150 --seconds 30 --out runs/detector.npz
```

**Watch the 1v1 model train live, in 3D:**
```bash
python -m sim.train_live --episodes 80 --out runs/detector.npz
```

**Run one 1v1 scenario and log it, for the HTML visualizer:**
```bash
python -m sim.run_scenario --attack 1 --detector runs/detector.npz --out runs/episode.json
```
`--attack`: 0=normal, 1=gps_spoof, 2=imu_corrupt, 3=actuator_injection, 4=comm_jam

**Watch a trained 1v1 scenario live in 3D** (no training, just playback with a fixed detector):
```bash
python -m sim.live_view --attack 1 --detector runs/detector.npz
```

**Record an actual MuJoCo-rendered GIF of a 1v1 scenario** (works headless, via EGL):
```bash
MUJOCO_GL=egl python -m sim.record_video --attack 1 --detector runs/detector.npz --out runs/demo.gif
```

**Open the 1v1 2D top-down visualizer**: open `two_drone_ids_visualizer.html` directly
in a browser — self-contained, no install, paused on load.

**Real ArduPilot bridge for the 1v1 scenario** (needs a real ArduCopter SITL build on
your machine — see `sim/ardupilot_bridge.py`'s module docstring for setup steps and the
unverified-against-real-firmware caveat):
```bash
python -m sim.ardupilot_bridge --attack 1
```

**Real ArduPilot bridge for the 2v2 scenario** (same setup requirement and caveat as
above — real ArduCopter SITL flies `b1`; `a1`/`a2`/`b2` fly in-sim, and `b2` runs live
detection + the Bayesian ejection decision + blockchain logging exactly as in
`train_live_four.py`). This version also fixes a gap in the original 1v1 bridge, which
loaded a detector but never actually called it during the loop:
```bash
python -m sim.ardupilot_bridge_four --attack 1
```

## 8. What's real vs. simplified — the honest summary

**Real, tested, and verified from a clean copy of this exact package before shipping:**
- The Skydio X2 MJCF is cloned from `google-deepmind/mujoco_menagerie`, not
  reconstructed — the actual mesh/texture files ship in `models/assets/`.
- The motor-mixing matrix is derived algebraically from the X2's real rotor geometry
  and yaw-torque gear coefficients in the MJCF, not hand-tuned.
- The REINFORCE training loop is genuine reward-driven weight updates against real
  MuJoCo physics.
- The Bayesian ejection module's transitory-vs-permanent behavior was directly tested
  with short/oscillating vs. sustained misbehavior sequences (Section 4).
- The blockchain ledger's tamper-evidence was directly tested by mutating committed
  block data, including a re-hashing attempt (Section 5).
- Formation flight tightness was measured directly (offset error of a few centimetres),
  not eyeballed.

**Simplified, on purpose, with the reasoning documented in-code and above:**
- `sim/controller.py` implements ArduCopter's real control *architecture*
  (position→velocity→attitude→rate→motor-mixer cascade) in numpy — it does not compile
  or run the actual ArduCopter firmware, which needs a large C++ build not practical in
  the environment this was built in. `sim/ardupilot_bridge.py` is a real,
  protocol-correct JSON-FDM bridge that would let real ArduCopter SITL fly the drone
  through this physics; its socket/JSON mechanics are tested via loopback, but it has
  **not** been verified end-to-end against real ArduCopter firmware.
- The detector is a 14-feature MLP, not the full CNN/GRU+attention architecture
  implied by a production PRHN system.
- The blockchain ledger has no real distributed consensus or P2P networking (Section 5).
- The Bayesian game's cost parameters are derived from this project's own measured
  detector performance, not an independently-sourced cost model (Section 4).

## 9. Known gaps

- `ardupilot_bridge_four.py`'s full loop (physics + PWM protocol + live detection +
  Bayesian ejection + blockchain) was tested end-to-end with a fake PWM-feeding thread
  standing in for real ArduCopter SITL — confirmed it runs to completion, correctly
  detects an attack, and correctly ejects with a valid blockchain commit. It has **not**
  been tested against real ArduCopter firmware, for the reason stated in Section 8.
- `runs/detector_four.npz` is a supervised-warm-start-only checkpoint (generated
  headless, the same way the original `detector.npz` was, to give a starting point) —
  it has not yet been through a live RL fine-tuning session, since that requires a
  display this environment doesn't have.
- Detection latency is real and unavoidable for gradually-ramping attacks: 1.25–2.17s
  for `gps_spoof` and `actuator_injection`, 0.06–0.43s for `imu_corrupt` and `comm_jam`
  (Section 3a). During the ramp-in window the attack genuinely isn't expressed in the
  telemetry yet, so this is a property of the attack model, not a detector deficiency —
  but it does mean raw per-class recall understates detection quality, which is why
  `sim/evaluate.py` reports latency and post-onset recall alongside it.
- Only `a1`/`b1` participate in the attack/detection loop — `a2` and `b2`'s
  escort/wingman roles are formation-flight only (`b2` additionally runs the detector
  on `b1`, but doesn't itself get attacked).

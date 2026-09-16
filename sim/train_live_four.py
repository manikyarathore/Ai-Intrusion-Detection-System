"""
Live 3D training for the 4-drone (2v2) scenario -- THE primary way to train in this
project. Every physics tick is a real mujoco.mj_step() call rendered live in MuJoCo's
viewer as it happens; there is no separate "train first, look at a log after" path for
this environment. (sim/train_rl.py / train_live.py, for the older 2-drone scenario,
still exist for reference but are not the recommended entry point going forward.)

Watch: b1 (blue) flies its patrol with b2 (cyan) in wingman formation; a1 (red) and a2
(orange) fly together on the other side; a1 attacks b1; b2's detector (being trained,
live) flags it; the Bayesian friend-monitor accumulates a Misbehavior Rate; once it's
been persistently over threshold, b2 proposes an ejection block, a2 co-signs, and the
block commits to the shared ledger -- all printed to the terminal as it happens.

Paused on load; press SPACE to start.

Usage:
    python -m sim.train_live_four --episodes 80 --out runs/detector_four.npz
"""
import argparse
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from .env_four import FourDroneIDSEnv
from .detector import PRHNLiteDetector
from .attacks import ATTACK_NAMES
from .train_rl import evaluate_recall  # reused as-is; works against any env with the
                                         # same reset()/step() contract, including this one

_paused = {"value": True}


def _key_callback(keycode):
    if keycode == 32:
        _paused["value"] = not _paused["value"]
        print("PLAYING" if not _paused["value"] else "PAUSED")


def _collect_calibration(env_factory, n_episodes_per_class=5, seed=999):
    """Same idea as calibrate.py's collect_calibration_features, adapted for
    FourDroneIDSEnv -- headless (fast; this is a one-time setup pass, not the training
    loop itself, so it isn't subject to the "must be live" requirement any more than
    loading a config file would be)."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for attack_id in [0, 1, 2, 3, 4]:
        for _ in range(n_episodes_per_class):
            s = int(rng.integers(0, 1_000_000))
            env = env_factory(s)
            onset = float(rng.uniform(0.25, 0.7)) if attack_id != 0 else 0.5
            obs, info = env.reset(force_attack_id=attack_id, onset_frac=onset, seed=s)
            feats.append(obs); labels.append(info["true_label"])
            terminated = truncated = False
            while not (terminated or truncated):
                obs, r, terminated, truncated, info = env.step(0)
                feats.append(obs); labels.append(info["true_label"])
    return np.array(feats), np.array(labels)


def train_live_four(n_episodes=80, max_episode_seconds=30.0, lr=0.004, seed=0,
                     out_path="runs/detector_four.npz", sync_every=1, eval_every=10):
    env_factory = lambda s: FourDroneIDSEnv(max_episode_seconds=max_episode_seconds, seed=s)
    env = env_factory(seed)
    detector = PRHNLiteDetector(seed=seed)

    print("Calibrating (headless, one-time setup)...")
    calib_feats, calib_labels = _collect_calibration(env_factory, n_episodes_per_class=5, seed=seed + 999)
    detector.calibrate(calib_feats)
    print("Supervised warm-start (headless, one-time setup)...")
    detector.supervised_pretrain(calib_feats, calib_labels, epochs=80, lr=0.1, seed=seed)

    def eval_fn(det):
        return evaluate_recall(det, seeds=range(6000, 6006), max_episode_seconds=max_episode_seconds,
                                env_factory=env_factory)

    best_score, best_state = -1.0, None
    def score_and_maybe_save():
        nonlocal best_score, best_state
        recalls, fpr = eval_fn(detector)
        score = min(recalls.values()) - 0.5 * fpr
        if score > best_score:
            best_score = score
            best_state = (detector.W1.copy(), detector.b1.copy(), detector.W2.copy(), detector.b2.copy())
        return recalls, fpr, score

    recalls, fpr, score = score_and_maybe_save()
    print(f"post-warm-start: recalls={ {k: round(v,2) for k,v in recalls.items()} } FPR={fpr:.3f} score={score:.3f}\n")

    rng = np.random.default_rng(seed)
    reward_baseline = 0.0

    print("Opening live 3D viewer (2v2 battlefield) -- SPACE to start training.\n")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=_key_callback) as viewer:
        step_counter, ep = 0, 0
        while ep < n_episodes and viewer.is_running():
            obs, info = env.reset(seed=seed + ep)
            ep_reward, n_correct, n_steps = 0.0, 0, 0
            terminated = truncated = False
            epsilon = max(0.02, 0.10 * (1 - ep / n_episodes))
            ejection_reported = False

            while not (terminated or truncated) and viewer.is_running():
                while _paused["value"] and viewer.is_running():
                    viewer.sync(); time.sleep(0.05)
                if not viewer.is_running():
                    break

                action, proba, cache = detector.act(obs, explore=True, epsilon=epsilon, rng=rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                detector.reinforce_update(cache, action, reward - reward_baseline)
                reward_baseline = 0.995 * reward_baseline + 0.005 * reward

                n_correct += int(action == info["true_label"])
                n_steps += 1
                ep_reward += reward
                obs = next_obs

                fs = info["friend_status"]
                if fs["ejected"] and not ejection_reported:
                    ejection_reported = True
                    print(f"   >> b2 proposed + a2 co-signed an EJECT block for b1 "
                          f"(MR={fs['mr']:.2f}) -- ledger length now {info['blockchain_len']}, "
                          f"valid={env.blockchain.is_valid()}")

                step_counter += 1
                if step_counter % sync_every == 0:
                    viewer.sync()

            if not viewer.is_running():
                print("viewer closed, stopping training early")
                break

            ep += 1
            acc = n_correct / max(n_steps, 1)
            print(f"[episode {ep:4d}/{n_episodes}] attack={ATTACK_NAMES[info['attack_id']]:20s} "
                  f"steps={n_steps:5d} reward={ep_reward:8.2f} step-acc={acc:5.1%}")

            if ep % eval_every == 0:
                recalls, fpr, score = score_and_maybe_save()
                tag = " <- new best" if score == best_score else ""
                print(f"   held-out eval: recalls={ {k: round(v,2) for k,v in recalls.items()} } "
                      f"FPR={fpr:.3f} score={score:.3f} (best={best_score:.3f}){tag}\n")

    if best_state is not None:
        detector.W1, detector.b1, detector.W2, detector.b2 = best_state
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    detector.save(out_path)
    print(f"\nSaved best checkpoint (score={best_score:.3f}) to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/detector_four.npz")
    p.add_argument("--sync-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    args = p.parse_args()
    train_live_four(n_episodes=args.episodes, max_episode_seconds=args.seconds, lr=args.lr,
                     seed=args.seed, out_path=args.out, sync_every=args.sync_every,
                     eval_every=args.eval_every)

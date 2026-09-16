"""
Live 3D training: the RL loop from train_rl.py, but running inside MuJoCo's native
viewer so you can watch the drones actually fly/attack/get detected episode-by-episode
as training happens, instead of training headless and only looking at a log afterward.

Paused on load; press SPACE to start. Episode number, current attack, reward, and
running accuracy print to the terminal live as training proceeds. Every `--eval-every`
episodes it also prints a full held-out per-attack recall breakdown, exactly like
train_rl.py, and keeps the best-scoring checkpoint (same validation-based selection).

Requires a display -- won't work over a headless SSH session without X forwarding.

Usage:
    python -m sim.train_live --episodes 80 --out runs/detector.npz
"""
import argparse
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from .env import TwoDroneIDSEnv
from .detector import PRHNLiteDetector
from .calibrate import collect_calibration_features
from .attacks import ATTACK_NAMES
from .train_rl import evaluate_recall

_paused = {"value": True}   # paused on load, per spec
_quit = {"value": False}


def _key_callback(keycode):
    if keycode == 32:  # space
        _paused["value"] = not _paused["value"]
        print("PLAYING" if not _paused["value"] else "PAUSED")


def train_live(n_episodes=80, max_episode_seconds=30.0, lr=0.004, seed=0,
               out_path="runs/detector.npz", sync_every=1, realtime=False, eval_every=10):
    env = TwoDroneIDSEnv(max_episode_seconds=max_episode_seconds, seed=seed)
    detector = PRHNLiteDetector(seed=seed)

    print("Calibrating features across normal + all 4 attacks (headless, fast)...")
    calib_feats, calib_labels = collect_calibration_features(
        n_episodes_per_class=6, max_episode_seconds=max_episode_seconds, seed=seed + 999, with_labels=True)
    detector.calibrate(calib_feats)
    print("Class-balanced supervised warm-start (headless, fast)...")
    detector.supervised_pretrain(calib_feats, calib_labels, epochs=80, lr=0.1, seed=seed)

    best_score = -1.0
    best_state = None

    def score_and_maybe_save():
        nonlocal best_score, best_state
        recalls, fpr = evaluate_recall(detector, seeds=range(6000, 6006), max_episode_seconds=max_episode_seconds)
        score = min(recalls.values()) - 0.5 * fpr
        if score > best_score:
            best_score = score
            best_state = (detector.W1.copy(), detector.b1.copy(), detector.W2.copy(), detector.b2.copy())
        return recalls, fpr, score

    recalls, fpr, score = score_and_maybe_save()
    print(f"post-warm-start checkpoint: recalls={ {k: round(v,2) for k,v in recalls.items()} }  "
          f"FPR={fpr:.3f}  score={score:.3f}\n")

    rng = np.random.default_rng(seed)
    reward_baseline = 0.0

    print("Opening live 3D viewer -- paused on load, press SPACE to start training.\n")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=_key_callback) as viewer:
        step_counter = 0
        ep = 0
        while ep < n_episodes and viewer.is_running():
            obs, info = env.reset(seed=seed + ep)
            ep_reward, n_correct, n_steps = 0.0, 0, 0
            terminated = truncated = False
            epsilon = max(0.02, 0.10 * (1 - ep / n_episodes))

            while not (terminated or truncated) and viewer.is_running():
                while _paused["value"] and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                if not viewer.is_running():
                    break

                step_start = time.time()
                action, proba, cache = detector.act(obs, explore=True, epsilon=epsilon, rng=rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                detector.reinforce_update(cache, action, reward - reward_baseline)
                reward_baseline = 0.995 * reward_baseline + 0.005 * reward

                n_correct += int(action == info["true_label"])
                n_steps += 1
                ep_reward += reward
                obs = next_obs

                step_counter += 1
                if step_counter % sync_every == 0:
                    viewer.sync()
                if realtime:
                    dt_left = env.dt - (time.time() - step_start)
                    if dt_left > 0:
                        time.sleep(dt_left)

            if not viewer.is_running():
                print("viewer closed, stopping training early")
                break

            ep += 1
            acc = n_correct / max(n_steps, 1)
            print(f"[episode {ep:4d}/{n_episodes}]  attack={ATTACK_NAMES[info['attack_id']]:20s} "
                  f"steps={n_steps:5d}  reward={ep_reward:8.2f}  step-acc={acc:5.1%}")

            if ep % eval_every == 0:
                recalls, fpr, score = score_and_maybe_save()
                tag = " <- new best" if score == best_score else ""
                print(f"   held-out eval: recalls={ {k: round(v,2) for k,v in recalls.items()} }  "
                      f"FPR={fpr:.3f}  score={score:.3f}  (best={best_score:.3f}){tag}\n")

    if best_state is not None:
        detector.W1, detector.b1, detector.W2, detector.b2 = best_state
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    detector.save(out_path)
    print(f"\nSaved best checkpoint (validation score={best_score:.3f}) to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/detector.npz")
    p.add_argument("--sync-every", type=int, default=1,
                    help="render every Nth physics step (higher = faster but choppier)")
    p.add_argument("--realtime", action="store_true",
                    help="throttle to real time (30s episodes take 30s); default runs as fast as rendering allows")
    p.add_argument("--eval-every", type=int, default=10)
    args = p.parse_args()

    train_live(n_episodes=args.episodes, max_episode_seconds=args.seconds, lr=args.lr, seed=args.seed,
               out_path=args.out, sync_every=args.sync_every, realtime=args.realtime, eval_every=args.eval_every)

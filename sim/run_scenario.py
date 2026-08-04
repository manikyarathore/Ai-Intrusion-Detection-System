"""
Runs one full main-vs-attacker scenario with a (trained or fresh) detector, logging
per-timestep telemetry, ground-truth attack labels, and detector predictions to JSON --
this is the file the paused-by-default HTML visualizer (Section 11.6) reads.

Usage:
    python -m sim.run_scenario --attack 1 --onset 0.35 --detector runs/detector.npz --out runs/episode.json
"""
import argparse
import json
import os

import numpy as np

from .env import TwoDroneIDSEnv
from .detector import PRHNLiteDetector
from .attacks import ATTACK_NAMES


def run(attack_id=1, onset_frac=0.35, x1y1=(-3.0, -3.0), x2y2=(3.0, 3.0),
        max_episode_seconds=14.0, detector_path=None, seed=7, decimate=2):
    env = TwoDroneIDSEnv(x1y1=x1y1, x2y2=x2y2, max_episode_seconds=max_episode_seconds, seed=seed)
    detector = PRHNLiteDetector(seed=seed)
    if detector_path and os.path.exists(detector_path):
        detector.load(detector_path)

    obs, info = env.reset(force_attack_id=attack_id, onset_frac=onset_frac, seed=seed, record=True)
    frames = []
    terminated = truncated = False
    step_i = 0
    while not (terminated or truncated):
        action, proba, z = detector.act(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        step_i += 1
        if step_i % decimate == 0:
            frames.append({
                "t": round(env._t, 3),
                "main": [round(v, 3) for v in info["main_pos"]],
                "att": env.get_log()[-1]["att_pos"],
                "true_label": info["true_label"],
                "pred_label": int(action),
                "range_m": round(env.get_log()[-1]["range_m"], 2),
            })

    out = {
        "x1y1": list(x1y1), "x2y2": list(x2y2),
        "attack_name": ATTACK_NAMES[attack_id],
        "attack_id": attack_id,
        "class_names": ATTACK_NAMES,
        "dt_frame": env.dt * decimate,
        "frames": frames,
    }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=int, default=1)
    p.add_argument("--onset", type=float, default=0.35)
    p.add_argument("--seconds", type=float, default=14.0)
    p.add_argument("--detector", type=str, default="runs/detector.npz")
    p.add_argument("--out", type=str, default="runs/episode.json")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    result = run(attack_id=args.attack, onset_frac=args.onset, max_episode_seconds=args.seconds,
                 detector_path=args.detector, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {len(result['frames'])} frames to {args.out}")

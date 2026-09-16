"""
Runs one full 2v2 scenario (FourDroneIDSEnv) with a trained detector, logging all 4
drone positions, ground-truth/predicted attack labels, and the live Bayesian
friend-monitor + blockchain ledger status per frame -- this is what the 2v2 HTML
visualizer reads.

Usage:
    python -m sim.run_scenario_four --attack 1 --onset 0.35 --detector runs/detector_four.npz --out runs/episode_four.json
"""
import argparse
import json
import os

from .env_four import FourDroneIDSEnv
from .detector import PRHNLiteDetector
from .attacks import ATTACK_NAMES


def run(attack_id=1, onset_frac=0.35, max_episode_seconds=30.0, detector_path=None,
        seed=7, decimate=2):
    env = FourDroneIDSEnv(max_episode_seconds=max_episode_seconds, seed=seed)
    detector = PRHNLiteDetector(seed=seed)
    if detector_path and os.path.exists(detector_path):
        detector.load(detector_path)

    obs, info = env.reset(force_attack_id=attack_id, onset_frac=onset_frac, seed=seed, record=True)
    frames = []
    terminated = truncated = False
    step_i = 0
    while not (terminated or truncated):
        action, proba, cache = detector.act(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        step_i += 1
        if step_i % decimate == 0:
            log_entry = env.get_log()[-1]
            fs = info["friend_status"]
            frames.append({
                "t": log_entry["t"],
                "a1": [round(v, 3) for v in log_entry["a1"]],
                "a2": [round(v, 3) for v in log_entry["a2"]],
                "b1": [round(v, 3) for v in log_entry["b1"]],
                "b2": [round(v, 3) for v in log_entry["b2"]],
                "true_label": info["true_label"],
                "pred_label": int(action),
                "mr": round(fs["mr"], 4),
                "monitoring_active": fs["monitoring_active"],
                "over_threshold_streak": fs["over_threshold_streak"],
                "ejected": fs["ejected"],
                "blockchain_len": info["blockchain_len"],
            })

    out = {
        "b1_start": list(env.start_xy), "b1_goal": list(env.goal_xy),
        "attack_name": ATTACK_NAMES[attack_id],
        "attack_id": attack_id,
        "class_names": ATTACK_NAMES,
        "dt_frame": env.dt * decimate,
        "b_attacker": env.friend_monitor.b_attacker,
        "b_node": env.friend_monitor.b_node,
        "persistence_ticks": env.friend_monitor.persistence_ticks,
        "blockchain_valid": env.blockchain.is_valid(),
        "frames": frames,
    }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=int, default=1)
    p.add_argument("--onset", type=float, default=0.35)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--detector", type=str, default="runs/detector_four.npz")
    p.add_argument("--out", type=str, default="runs/episode_four.json")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    result = run(attack_id=args.attack, onset_frac=args.onset, max_episode_seconds=args.seconds,
                 detector_path=args.detector, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {len(result['frames'])} frames to {args.out}")

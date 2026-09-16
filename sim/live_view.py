"""
Live, interactive 3D view of the battlefield scenario using MuJoCo's native viewer.
This is the real MuJoCo simulation environment (models/battlefield_scene.xml) running
live -- not a log playback. Paused on load; press SPACE to play/pause, matching Section
11.6's "must not auto-play" requirement, just at the native-viewer level instead of the
HTML visualizer.

Requires a display (this won't work over a headless SSH session without X forwarding /
a virtual display). Run from inside the project root:

    python -m sim.live_view --attack 1 --detector runs/detector.npz
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from .env import TwoDroneIDSEnv
from .detector import PRHNLiteDetector
from .attacks import ATTACK_NAMES

_paused = {"value": True}   # paused on load, per spec


def _key_callback(keycode):
    if chr(keycode) == " " or keycode == 32:
        _paused["value"] = not _paused["value"]
        print("PLAYING" if not _paused["value"] else "PAUSED")


def main(attack_id=1, onset_frac=0.4, detector_path="runs/detector.npz",
         max_episode_seconds=30.0, seed=7, realtime=True):
    env = TwoDroneIDSEnv(max_episode_seconds=max_episode_seconds, seed=seed)
    detector = PRHNLiteDetector(seed=seed)
    try:
        detector.load(detector_path)
        print(f"loaded trained detector from {detector_path}")
    except Exception as e:
        print(f"no trained detector loaded ({e}); using an untrained one")

    obs, info = env.reset(force_attack_id=attack_id, onset_frac=onset_frac, seed=seed)
    print(f"Scenario: {ATTACK_NAMES[attack_id]}  |  SPACE to play/pause  |  paused on load")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=_key_callback) as viewer:
        terminated = truncated = False
        last_print = 0.0
        while viewer.is_running() and not (terminated or truncated):
            step_start = time.time()
            if not _paused["value"]:
                action, proba, cache = detector.act(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(action)
                if env._t - last_print > 1.0:
                    last_print = env._t
                    match = "OK" if action == info["true_label"] else "MISS"
                    print(f"t={env._t:5.1f}s  true={ATTACK_NAMES[info['true_label']]:20s} "
                          f"pred={ATTACK_NAMES[action]:20s} [{match}]")
            viewer.sync()
            if realtime:
                dt_left = env.dt - (time.time() - step_start)
                if dt_left > 0:
                    time.sleep(dt_left)

    print("done" if terminated or truncated else "viewer closed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=int, default=1)
    p.add_argument("--onset", type=float, default=0.4)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--detector", type=str, default="runs/detector.npz")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    main(attack_id=args.attack, onset_frac=args.onset, detector_path=args.detector,
         max_episode_seconds=args.seconds, seed=args.seed)

"""
Records a real GIF of a 2v2 scenario using MuJoCo's offscreen renderer.

Usage:
    MUJOCO_GL=egl python -m sim.record_video_four --attack 1 --detector runs/detector_four.npz --out runs/scenario_four.gif
"""
import argparse
import os

import mujoco
import PIL.Image

from .env_four import FourDroneIDSEnv
from .detector import PRHNLiteDetector


def record(attack_id=1, onset_frac=0.4, detector_path="runs/detector_four.npz",
           max_episode_seconds=30.0, seed=7, out_path="runs/scenario_four.gif",
           fps=15, width=960, height=540, cam_lookat=(1, -6, 1), cam_distance=42,
           cam_azimuth=-55, cam_elevation=-25, max_frames=240):
    env = FourDroneIDSEnv(max_episode_seconds=max_episode_seconds, seed=seed)
    detector = PRHNLiteDetector(seed=seed)
    try:
        detector.load(detector_path)
    except Exception as e:
        print(f"no trained detector ({e}); using an untrained one")

    obs, info = env.reset(force_attack_id=attack_id, onset_frac=onset_frac, seed=seed)

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat = list(cam_lookat)
    cam.distance = cam_distance
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation

    frames = []
    steps_per_frame = max(1, int(round(1.0 / (fps * env.dt))))
    terminated = truncated = False
    step_i = 0
    ejected_at_frame = None

    while not (terminated or truncated) and len(frames) < max_frames:
        action, proba, cache = detector.act(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        step_i += 1
        if step_i % steps_per_frame == 0:
            renderer.update_scene(env.data, camera=cam)
            img = renderer.render()
            frames.append(PIL.Image.fromarray(img))
            if info["friend_status"]["ejected"] and ejected_at_frame is None:
                ejected_at_frame = len(frames) - 1

    if not frames:
        print("no frames captured")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                    duration=int(1000 / fps), loop=0)
    print(f"wrote {len(frames)} frames ({len(frames)/fps:.1f}s @ {fps}fps) to {out_path}"
          + (f" -- ejection visible at frame {ejected_at_frame}" if ejected_at_frame else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=int, default=1)
    p.add_argument("--onset", type=float, default=0.4)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--detector", type=str, default="runs/detector_four.npz")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="runs/scenario_four.gif")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--max-frames", type=int, default=240)
    args = p.parse_args()

    record(attack_id=args.attack, onset_frac=args.onset, detector_path=args.detector,
           max_episode_seconds=args.seconds, seed=args.seed, out_path=args.out,
           fps=args.fps, max_frames=args.max_frames)

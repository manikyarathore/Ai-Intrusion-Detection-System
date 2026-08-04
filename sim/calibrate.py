"""
Collects a representative batch of physics-residual feature vectors (normal flight +
all 4 attacks, several onset times) and uses it to calibrate (and freeze) the
detector's feature normalization -- see PRHNLiteDetector.calibrate() for why this
replaces an ever-adapting online normalizer.
"""
import numpy as np
from .env import TwoDroneIDSEnv


def collect_calibration_features(n_episodes_per_class=6, max_episode_seconds=14.0, seed=123,
                                  with_labels=False):
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for attack_id in [0, 1, 2, 3, 4]:
        for i in range(n_episodes_per_class):
            s = int(rng.integers(0, 1_000_000))
            env = TwoDroneIDSEnv(max_episode_seconds=max_episode_seconds, seed=s)
            onset = float(rng.uniform(0.25, 0.7)) if attack_id != 0 else 0.5
            obs, info = env.reset(force_attack_id=attack_id, onset_frac=onset, seed=s)
            feats.append(obs)
            labels.append(info["true_label"])
            terminated = truncated = False
            while not (terminated or truncated):
                obs, r, terminated, truncated, info = env.step(0)  # action doesn't affect dynamics
                feats.append(obs)
                labels.append(info["true_label"])
    feats = np.array(feats)
    if with_labels:
        return feats, np.array(labels)
    return feats

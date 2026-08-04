"""
Section 11.5's RL training loop: trains the PRHN-lite detector's classification head
online, via REINFORCE, using the reward/penalty structure defined in env.py.

Usage:
    python -m sim.train_rl --episodes 300 --out runs/detector.npz
"""
import argparse
import copy
import time
import numpy as np

from .env import TwoDroneIDSEnv
from .detector import PRHNLiteDetector
from .calibrate import collect_calibration_features


def evaluate_recall(detector, seeds=range(5000, 5010), max_episode_seconds=14.0, onset_frac=0.4):
    """Held-out per-class recall + normal-phase false-positive rate, used to pick the
    best checkpoint during RL fine-tuning (Section 11.5 addendum)."""
    from .attacks import ATTACK_NAMES
    recalls = {}
    fp_total, fp_count = 0, 0
    for atk in range(1, 5):
        correct = total = 0
        for s in seeds:
            env = TwoDroneIDSEnv(max_episode_seconds=max_episode_seconds, seed=s)
            obs, info = env.reset(force_attack_id=atk, onset_frac=onset_frac, seed=s)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _, _ = detector.act(obs, explore=False)
                obs, r, terminated, truncated, info = env.step(action)
                if info["true_label"] == atk:
                    total += 1
                    correct += int(action == atk)
                elif info["true_label"] == 0:
                    fp_count += 1
                    fp_total += int(action != 0)
        recalls[ATTACK_NAMES[atk]] = correct / max(total, 1)
    fpr = fp_total / max(fp_count, 1)
    return recalls, fpr


def train(n_episodes=300, max_episode_seconds=12.0, lr=0.05, leave_one_out_id=None,
          eval_every=25, seed=0, log_path=None):
    env = TwoDroneIDSEnv(max_episode_seconds=max_episode_seconds,
                          leave_one_out_id=leave_one_out_id, seed=seed)
    detector = PRHNLiteDetector(lr=lr, seed=seed)

    # one-time calibration (Section 11.5 addendum): freeze normalization from a
    # representative sample of normal + attack traffic before any RL updates happen,
    # so a sustained attack can't get gradually normalized away mid-episode.
    calib_feats, calib_labels = collect_calibration_features(
        n_episodes_per_class=6, seed=seed + 999, with_labels=True)
    detector.calibrate(calib_feats)
    # class-balanced supervised warm-start, then RL fine-tunes from there (see
    # PRHNLiteDetector.supervised_pretrain for why this avoids REINFORCE's cold-start
    # exploration collapse on the rarer attack classes)
    detector.supervised_pretrain(calib_feats, calib_labels, epochs=80, lr=0.3, seed=seed)

    rng = np.random.default_rng(seed)
    reward_baseline = 0.0  # running mean reward, subtracted as a variance-reduction
                            # baseline (standard REINFORCE trick) -- raw reward as the
                            # gradient scale was noisy enough to overwrite the
                            # supervised warm-start's already-good decision boundary
                            # for the rarer attack classes

    reward_history = []
    accuracy_history = []
    t0 = time.time()

    # validation-based checkpoint selection: score = min per-class recall (not the mean --
    # a checkpoint that aces 3 classes and completely drops a 4th should score worse than
    # one that's moderately good at all 4) minus a penalty for false positives.
    best_score = -1.0
    best_state = None

    def score_and_maybe_save(detector):
        nonlocal best_score, best_state
        recalls, fpr = evaluate_recall(detector, seeds=range(6000, 6006))
        score = min(recalls.values()) - 0.5 * fpr
        if score > best_score:
            best_score = score
            best_state = (detector.W.copy(), detector.b.copy())
        return recalls, fpr, score

    # score the post-supervised-pretrain checkpoint too, so RL fine-tuning can only ever
    # replace it if it actually finds something better on held-out data
    score_and_maybe_save(detector)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward, n_correct, n_steps = 0.0, 0, 0
        terminated = truncated = False

        # with a supervised warm-start already in place, RL fine-tuning needs much less
        # forced exploration than a cold start would -- just enough to keep adapting
        epsilon = max(0.02, 0.10 * (1 - ep / n_episodes))

        while not (terminated or truncated):
            action, proba, z = detector.act(obs, explore=True, epsilon=epsilon, rng=rng)
            next_obs, reward, terminated, truncated, info = env.step(action)

            detector.reinforce_update(z, action, reward - reward_baseline)
            reward_baseline = 0.995 * reward_baseline + 0.005 * reward

            correct = int(action == info["true_label"]) or (
                leave_one_out_id is not None
                and info["true_label"] == leave_one_out_id
                and action != 0
            )
            n_correct += correct
            n_steps += 1
            ep_reward += reward
            obs = next_obs

        reward_history.append(ep_reward / max(n_steps, 1))  # mean reward per step
        accuracy_history.append(n_correct / max(n_steps, 1))

        if (ep + 1) % eval_every == 0:
            recent_r = np.mean(reward_history[-eval_every:])
            recent_a = np.mean(accuracy_history[-eval_every:])
            recalls, fpr, score = score_and_maybe_save(detector)
            print(f"episode {ep+1:4d}/{n_episodes} | "
                  f"mean reward (last {eval_every}) = {recent_r:8.2f} | "
                  f"mean step-accuracy = {recent_a:5.1%} | "
                  f"held-out recalls = {{{', '.join(f'{k}:{v:.2f}' for k,v in recalls.items())}}} | "
                  f"FPR={fpr:.3f} | checkpoint score={score:.3f} (best={best_score:.3f}) | "
                  f"elapsed {time.time()-t0:5.1f}s")

    # restore the best-scoring checkpoint seen (by held-out min-class-recall), not
    # necessarily the final weights -- see the note above evaluate_recall()
    if best_state is not None:
        detector.W, detector.b = best_state

    if log_path:
        detector.save(log_path)
        np.savez(log_path.replace(".npz", "_curves.npz"),
                 reward=np.array(reward_history), accuracy=np.array(accuracy_history))
    return detector, reward_history, accuracy_history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leave-one-out", type=int, default=None,
                    help="attack id (1-4) to withhold from training, for the LOAO open-set test")
    p.add_argument("--out", type=str, default="runs/detector.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train(n_episodes=args.episodes, max_episode_seconds=args.seconds, lr=args.lr,
          leave_one_out_id=args.leave_one_out, seed=args.seed, log_path=args.out)

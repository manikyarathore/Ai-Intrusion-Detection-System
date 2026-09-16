"""
Evaluation metrics for the detector that separate *detection quality* from *detection
latency* -- which raw per-class recall conflates, sometimes badly.

Why this module exists: raw recall counts every attack-active tick equally, including
the ticks right after onset when a gradually-ramping attack is still physically
indistinguishable from normal flight. That makes an attack's recall depend heavily on
how long its episodes happen to run: `actuator_injection` scored ~67% recall vs
`gps_spoof`'s ~90%, which looks like a much worse detector -- but measuring latency
directly showed actuator_injection is actually detected FASTER (1.25s vs 2.17s after
onset). Its recall is lower only because it frequently crashes the drone and ends the
episode early, so the unavoidable ramp-in window is a larger fraction of a shorter
episode.

So this module reports three things instead of one:
  - detection_latency: seconds from attack onset to first correct classification
  - post_onset_recall: recall computed only over ticks at least `settle_seconds` after
    onset -- i.e. "once the attack is actually expressed in the telemetry, do we catch it?"
  - raw_recall: the original metric, kept for comparability with earlier results

None of these is "the" right number on its own. Latency matters for a real IDS (how fast
do you know?); post-onset recall matters for sustained detection (do you stay on it?);
raw recall is what the training reward optimizes. Reporting all three avoids the trap of
reading a single number as "this attack is detected worse."
"""
import numpy as np

from .attacks import ATTACK_NAMES


def evaluate_detailed(detector, env_factory, seeds=range(2000, 2010), onset_frac=0.4,
                       settle_seconds=2.0):
    """Returns a dict per attack with raw_recall, post_onset_recall, detection_latency
    (mean seconds, or None if never detected in a run), and normal-phase FPR."""
    results = {}
    fp_total = fp_count = 0

    for atk in range(1, 5):
        raw_correct = raw_total = 0
        post_correct = post_total = 0
        latencies = []

        for s in seeds:
            env = env_factory(s)
            obs, info = env.reset(force_attack_id=atk, onset_frac=onset_frac, seed=s)
            settle_ticks = int(settle_seconds / env.dt)
            terminated = truncated = False
            onset_tick = detect_tick = None
            i = 0

            while not (terminated or truncated):
                action, _, _ = detector.act(obs, explore=False)
                obs, r, terminated, truncated, info = env.step(action)
                i += 1
                if info["true_label"] == atk:
                    if onset_tick is None:
                        onset_tick = i
                    if action == atk and detect_tick is None:
                        detect_tick = i
                    raw_total += 1
                    raw_correct += int(action == atk)
                    if i - onset_tick >= settle_ticks:
                        post_total += 1
                        post_correct += int(action == atk)
                elif info["true_label"] == 0:
                    fp_count += 1
                    fp_total += int(action != 0)

            if detect_tick is not None and onset_tick is not None:
                latencies.append((detect_tick - onset_tick) * env.dt)

        results[ATTACK_NAMES[atk]] = {
            "raw_recall": raw_correct / max(raw_total, 1),
            "post_onset_recall": post_correct / max(post_total, 1) if post_total else None,
            "detection_latency_s": float(np.mean(latencies)) if latencies else None,
            "detected_in_n_runs": f"{len(latencies)}/{len(list(seeds))}",
        }

    return results, fp_total / max(fp_count, 1)


def print_report(results, fpr, settle_seconds=2.0):
    print(f"{'attack':22s} {'raw recall':>11s} {'post-onset':>11s} {'latency':>9s}  detected")
    print(f"{'':22s} {'':>11s} {'recall':>11s} {'(s)':>9s}")
    print("-" * 68)
    for name, r in results.items():
        post = f"{r['post_onset_recall']:.3f}" if r["post_onset_recall"] is not None else "n/a"
        lat = f"{r['detection_latency_s']:.2f}" if r["detection_latency_s"] is not None else "never"
        print(f"{name:22s} {r['raw_recall']:11.3f} {post:>11s} {lat:>9s}  {r['detected_in_n_runs']}")
    print("-" * 68)
    print(f"normal-phase false-positive rate: {fpr:.4f}")
    print(f"(post-onset recall measured from {settle_seconds}s after attack onset)")

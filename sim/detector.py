"""
A lightweight stand-in for the full PRHN architecture in Section 5 of the framework:

  - Physics branch: a closed-form rigid-body prediction of next-state telemetry from
    current state + commanded motor ctrl (no training needed, exactly as Section 5.3a
    describes as the minimum-viable physics branch).
  - Data/residual branch: the discrepancy between predicted and reported telemetry,
    turned into a compact feature vector.
  - Classification head: a single linear layer + softmax over
    {normal, gps_spoof, imu_corrupt, actuator_injection, comm_jam} -- deliberately the
    simplest possible data branch (Section 5.3a's data branch can be "a small CNN/GRU";
    this is the lightest version of that, sized so the RL update in Section 11.5 is a
    plain, auditable REINFORCE step rather than requiring a deep-learning framework).

Swapping this classification head for the CNN/GRU + attention layer in Section 5.3a is
a drop-in change -- `featurize()` is the interface boundary.
"""
import numpy as np

from .controller import MASS, GRAVITY
from .attacks import ATTACK_NAMES

N_CLASSES = len(ATTACK_NAMES)
N_FEATURES = 9


def physics_predict_next_pos(pos, vel, dt):
    """Closed-form constant-velocity prediction (the minimum-viable physics branch --
    Section 5.3a). A higher-fidelity version would integrate full rigid-body dynamics
    from the commanded ctrl; this is intentionally the lightweight variant."""
    return pos + vel * dt


def featurize(window):
    """window: dict of recent-history arrays (see env.py) -> fixed-size feature vector."""
    gps_residual = np.linalg.norm(window["gps_reported"][-1] - window["gps_physics_pred"][-1])
    gyro_mag = np.linalg.norm(window["gyro"][-1])
    gyro_jump = np.linalg.norm(window["gyro"][-1] - window["gyro"][-2]) if len(window["gyro"]) > 1 else 0.0
    accel_mag = np.linalg.norm(window["accel"][-1] - np.array([0, 0, GRAVITY]))
    ctrl_vs_frc_residual = np.linalg.norm(window["ctrl_cmd"][-1] - window["ctrl_applied"][-1])
    comm_gap = window["comm_gap_steps"][-1]
    dropped_list = list(window["comm_dropped"])[-10:]
    comm_drop_rate = float(np.mean(dropped_list)) if dropped_list else 0.0
    mag_residual = float(np.linalg.norm(window["mag"][-1] - window["mag"][0])) if len(window["mag"]) else 0.0
    speed = float(np.linalg.norm(window["vel"][-1]))
    return np.array([
        gps_residual, gyro_mag, gyro_jump, accel_mag, ctrl_vs_frc_residual,
        comm_gap, comm_drop_rate, mag_residual, speed,
    ])


class PRHNLiteDetector:
    """Linear-softmax classification head over physics-residual features, trained
    online via REINFORCE (Section 11.5)."""

    def __init__(self, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.05, size=(N_FEATURES, N_CLASSES))
        self.b = np.zeros(N_CLASSES)
        self.lr = lr
        self._feat_mean = np.zeros(N_FEATURES)
        self._feat_std = np.ones(N_FEATURES)
        self._n_seen = 0
        self._frozen = False

    def calibrate(self, feature_samples):
        """Set FIXED normalization stats from a representative batch of feature vectors
        (collected once, up front, across normal + attack conditions -- see
        sim/calibrate.py) and freeze them. This replaces an earlier design where the
        normalizer kept adapting online during training/deployment: for a *sustained*
        attack (e.g. a slowly ramping GPS spoof), an ever-adapting baseline gradually
        re-absorbs the elevated residual as "the new normal," silently erasing the
        detector's own sensitivity over the course of the attack it's supposed to be
        catching. A one-time calibration avoids that feedback loop."""
        feature_samples = np.asarray(feature_samples)
        self._feat_mean = feature_samples.mean(axis=0)
        self._feat_std = feature_samples.std(axis=0)
        self._frozen = True

    def _normalize(self, feat):
        if self._frozen:
            return (feat - self._feat_mean) / (self._feat_std + 1e-3)
        # (fallback, pre-calibration) simple running normalization
        self._n_seen += 1
        alpha = 1.0 / min(self._n_seen, 2000)
        self._feat_mean = (1 - alpha) * self._feat_mean + alpha * feat
        self._feat_std = (1 - alpha) * self._feat_std + alpha * np.abs(feat - self._feat_mean)
        return (feat - self._feat_mean) / (self._feat_std + 1e-3)

    def predict_proba(self, feat):
        z = self._normalize(feat)
        logits = z @ self.W + self.b
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return p, z

    def act(self, feat, explore=True, epsilon=0.0, rng=None):
        p, z = self.predict_proba(feat)
        rng = rng or np.random
        if explore:
            if epsilon > 0 and rng.random() < epsilon:
                # forced uniform exploration -- prevents REINFORCE from collapsing onto
                # whichever classes got lucky/clean gradient signal early and starving
                # the others of any further exploration (a standard failure mode of
                # vanilla policy-gradient without an entropy/exploration floor)
                action = int(rng.integers(N_CLASSES))
            else:
                action = rng.choice(N_CLASSES, p=p)
        else:
            action = int(np.argmax(p))
        return action, p, z

    def reinforce_update(self, z, action, reward):
        """Standard softmax policy-gradient step: W += lr * reward * z (outer) (onehot(a) - p)."""
        logits = z @ self.W + self.b
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        onehot = np.zeros(N_CLASSES)
        onehot[action] = 1.0
        grad = np.outer(z, onehot - p)  # gradient of log pi(a|s) w.r.t. W
        self.W += self.lr * reward * grad
        self.b += self.lr * reward * (onehot - p)

    def supervised_pretrain(self, features, labels, epochs=60, lr=0.3, seed=0):
        """Warm-start the linear-softmax weights with plain class-balanced cross-entropy
        on the (already-calibrated/frozen-normalized) feature batch, before Section
        11.5's RL fine-tuning takes over. Pure REINFORCE from a random init has to
        stumble onto a rare class's correct action before it gets any gradient signal
        for it at all; with 5 classes and most timesteps being "normal", that cold start
        reliably starves the rarer attack classes. A cheap supervised warm-start (this
        method) avoids that without changing anything about the RL loop itself -- it's
        still the RL reward signal that does the fine-tuning after this."""
        assert self._frozen, "call calibrate() before supervised_pretrain()"
        rng = np.random.default_rng(seed)
        Z = np.array([self._normalize_frozen(f) for f in features])
        y = np.asarray(labels)

        # class-balanced minibatches: undersample the dominant "normal" class each epoch
        idx_by_class = [np.where(y == c)[0] for c in range(N_CLASSES)]
        min_count = min(len(idx) for idx in idx_by_class if len(idx) > 0)
        min_count = max(min_count, 8)

        for _ in range(epochs):
            batch_idx = np.concatenate([
                rng.choice(idx, size=min(min_count, len(idx)), replace=len(idx) < min_count)
                for idx in idx_by_class if len(idx) > 0
            ])
            rng.shuffle(batch_idx)
            Zb, yb = Z[batch_idx], y[batch_idx]

            logits = Zb @ self.W + self.b
            logits -= logits.max(axis=1, keepdims=True)
            expv = np.exp(logits)
            P = expv / expv.sum(axis=1, keepdims=True)
            Y = np.zeros_like(P)
            Y[np.arange(len(yb)), yb] = 1.0

            grad_logits = (P - Y) / len(yb)
            grad_W = Zb.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)
            self.W -= lr * grad_W
            self.b -= lr * grad_b

        # the batches above were class-balanced (equal counts per class) so the rare
        # attack classes actually get gradient signal, but that means the trained
        # decision boundary implicitly assumes a uniform class prior. Real flight is
        # overwhelmingly "normal", so correct the bias back to the *natural* class
        # frequencies in the calibration set (standard prior-correction for training
        # on a class-balanced sample and deploying under the true, imbalanced one).
        natural_freq = np.array([max(np.mean(y == c), 1e-4) for c in range(N_CLASSES)])
        correction_strength = 0.5  # partial correction: full correction (1.0) restores a
                                    # perfectly-calibrated natural prior but, combined with
                                    # how rare/brief the attack classes are, ends up crushing
                                    # recall on them; 0.5 is a deliberate recall/FPR trade-off
        self.b += correction_strength * np.log(natural_freq * N_CLASSES)

    def _normalize_frozen(self, feat):
        return (feat - self._feat_mean) / (self._feat_std + 1e-3)

    def save(self, path):
        np.savez(path, W=self.W, b=self.b, feat_mean=self._feat_mean, feat_std=self._feat_std,
                  frozen=self._frozen)

    def load(self, path):
        d = np.load(path)
        self.W, self.b = d["W"], d["b"]
        self._feat_mean, self._feat_std = d["feat_mean"], d["feat_std"]
        self._frozen = bool(d["frozen"]) if "frozen" in d else True

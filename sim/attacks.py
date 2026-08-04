"""
The 4 attack types from Section 5 of the framework, implemented as pure functions
that corrupt exactly the sensor/actuator channel each attack targets. Parameters are
kept in physically-plausible ranges (small drift rates / bias magnitudes) as called
for in Section 10's "Attack realism" risk mitigation.

Attack IDs: 0=normal, 1=gps_spoof, 2=imu_corrupt, 3=actuator_injection, 4=comm_jam
"""
import numpy as np

ATTACK_NAMES = ["normal", "gps_spoof", "imu_corrupt", "actuator_injection", "comm_jam"]


class AttackState:
    """Holds the mutable state needed across timesteps for ramping/drifting attacks."""

    def __init__(self):
        self.active_id = 0
        self.onset_time = None
        self.gps_drift = np.zeros(3)
        self.severity = 0.0

    def trigger(self, attack_id, t):
        self.active_id = attack_id
        self.onset_time = t
        self.gps_drift[:] = 0
        self.severity = 0.0

    def clear(self):
        self.active_id = 0
        self.onset_time = None
        self.severity = 0.0

    def time_since_onset(self, t):
        return 0.0 if self.onset_time is None else max(0.0, t - self.onset_time)


def apply_gps_spoof(true_pos, state: AttackState, t, drift_rate=0.15, max_offset=3.0):
    """Attack 1: gradually drifting GNSS offset. Returns the *reported* (spoofed) position."""
    dt_since = state.time_since_onset(t)
    state.severity = min(1.0, dt_since / 4.0)
    # a fixed drift direction, ramping in magnitude -- a step-function variant is just drift_rate -> inf
    direction = np.array([1.0, 0.6, 0.0])
    direction /= np.linalg.norm(direction)
    offset = np.minimum(drift_rate * dt_since, max_offset) * direction
    state.gps_drift = offset
    return true_pos + offset


def apply_imu_corrupt(gyro, accel, state: AttackState, t, bias_rate=0.5, spike_prob=0.05):
    """Attack 2: increasing bias + occasional spikes on gyro/accel."""
    dt_since = state.time_since_onset(t)
    state.severity = min(1.0, dt_since / 3.0)
    bias = bias_rate * state.severity
    g = gyro + np.array([bias, -0.5 * bias, 0.3 * bias])
    a = accel + np.array([0.0, 0.0, 2.0 * bias])
    if np.random.rand() < spike_prob:
        spike = np.random.uniform(2.0, 5.0, size=3) * np.random.choice([-1, 1], 3)
        g = g + spike
    return g, a


def apply_actuator_injection(ctrl, state: AttackState, t, magnitude=4.0, motor_idx=0):
    """Attack 3: false thrust command injected on one motor (proxy for firmware compromise / MITM)."""
    dt_since = state.time_since_onset(t)
    state.severity = min(1.0, dt_since / 2.0)
    corrupted = ctrl.copy()
    corrupted[motor_idx] = np.clip(corrupted[motor_idx] + magnitude * state.severity, 0.0, 13.0)
    return corrupted


def apply_comm_jam(state: AttackState, t, base_latency_steps=0, max_extra_latency=15, max_drop_prob=0.6):
    """Attack 4: returns (extra_latency_steps, drop_probability) for comm_link.py to apply,
    scaled by how long the jam has been active (an escalating jammer, not an instant kill)."""
    dt_since = state.time_since_onset(t)
    state.severity = min(1.0, dt_since / 2.5)
    extra_latency = int(base_latency_steps + max_extra_latency * state.severity)
    drop_prob = max_drop_prob * state.severity
    return extra_latency, drop_prob


def label_for(state: AttackState):
    return state.active_id if state.active_id else 0

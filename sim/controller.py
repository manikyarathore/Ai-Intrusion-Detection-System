"""
ArduPilot-style cascaded flight controller for the Skydio X2 model.

ArduCopter's real control stack is a cascade of loops:
    Position (P)  ->  Velocity (PID)  ->  Attitude (P)  ->  Rate (PID)  ->  Motor mixer

This module implements that same cascade in numpy (no ArduPilot binary is compiled/run
here -- see README.md for why -- but the *control architecture* is the real ArduCopter
architecture, not a generic PID/LQR blob). The motor-mixing matrix is derived exactly
from the Skydio X2's rotor positions and yaw-torque gains as defined in
two_drone_scene.xml, not hand-tuned.
"""
import numpy as np


def rotate_body_to_world(quat, v):
    """Rotate vector v from body frame to world frame using MuJoCo's (w,x,y,z) quat."""
    w, x, y, z = quat
    qv = np.array([x, y, z])
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2 * (w * uv + uuv)


def quat_to_euler(q):
    """MuJoCo quat convention: (w, x, y, z) -> roll, pitch, yaw (rad)."""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


# ---- motor mixing matrix, derived exactly from two_drone_scene.xml geometry ----
# rotor (x, y) offsets and yaw-torque gear coefficients, copied from the MJCF <site>/<motor> defs
_ROTOR_XY = np.array([
    [-.14, -.18],   # thrust1
    [-.14,  .18],   # thrust2
    [ .14,  .18],   # thrust3
    [ .14, -.18],   # thrust4
])
_ROTOR_YAW_GAIN = np.array([-.0201, .0201, -.0201, .0201])

def _build_allocation_matrix():
    # rows: [Fz, tau_x(roll), tau_y(pitch), tau_z(yaw)] ; columns: rotor 1..4
    A = np.zeros((4, 4))
    A[0, :] = 1.0                              # each rotor contributes 1x to vertical thrust
    A[1, :] = _ROTOR_XY[:, 1]                  # tau_x = sum(ry_i * f_i)   (r x F, F=(0,0,f))
    A[2, :] = -_ROTOR_XY[:, 0]                 # tau_y = sum(-rx_i * f_i)
    A[3, :] = _ROTOR_YAW_GAIN                  # tau_z = sum(gear_z_i * f_i), reaction torque
    return A

_ALLOC = _build_allocation_matrix()
_ALLOC_INV = np.linalg.inv(_ALLOC)

MASS = 1.325                  # true simulated mass (verified against m.body_subtreemass in MuJoCo,
                               # not hand-summed -- an earlier hand-summed version double-counted
                               # the rotor masses and used 2.325, causing ~1.75x thrust overcommand)
GRAVITY = 9.81


class ArduCopterStyleController:
    """One instance per drone. Call update() every physics tick."""

    def __init__(self, dt=0.01, max_speed=2.0, max_tilt=0.35, ctrl_limit=13.0):
        self.dt = dt
        self.max_speed = max_speed
        self.max_tilt = max_tilt          # radians, ~20 deg
        self.ctrl_limit = ctrl_limit

        # gains modeled after ArduCopter's default pos/vel/att/rate loop structure
        self.kp_pos = 1.0
        self.kp_vel, self.ki_vel, self.kd_vel = 3.0, 0.4, 0.15
        self.kp_att = 6.0
        self.kp_rate, self.ki_rate, self.kd_rate = 0.15, 0.02, 0.01

        self._vel_i = np.zeros(3)
        self._rate_i = np.zeros(3)
        self._prev_rate_err = np.zeros(3)

    def reset(self):
        self._vel_i[:] = 0
        self._rate_i[:] = 0
        self._prev_rate_err[:] = 0

    def update(self, pos, vel, quat, gyro, pos_target, yaw_target=0.0, vel_ff=None):
        """Returns 4 motor ctrl values (clipped to [0, ctrl_limit]).
        vel_ff: optional velocity feed-forward (e.g. a moving target's own velocity),
        so tracking a moving waypoint doesn't carry a permanent steady-state lag."""
        # ---- Position loop (P) -> desired velocity ----
        pos_err = pos_target - pos
        ff = vel_ff if vel_ff is not None else np.zeros(3)
        vel_des = np.clip(self.kp_pos * pos_err + ff, -self.max_speed, self.max_speed)

        # ---- Velocity loop (PID) -> desired acceleration ----
        vel_err = vel_des - vel
        self._vel_i += vel_err * self.dt
        self._vel_i = np.clip(self._vel_i, -2.0, 2.0)  # anti-windup: a persistent fault
                                                          # (e.g. a stuck/corrupted motor)
                                                          # never lets vel_err resolve, so
                                                          # an unclamped integral eventually
                                                          # winds up and diverges over a
                                                          # long episode
        acc_des = self.kp_vel * vel_err + self.ki_vel * self._vel_i - self.kd_vel * vel
        acc_des[2] += GRAVITY  # gravity feed-forward for hover

        # ---- Desired attitude from desired acceleration (small-angle inversion) ----
        thrust_des = MASS * np.linalg.norm([acc_des[0], acc_des[1], acc_des[2]])
        thrust_des = max(thrust_des, 0.1)
        roll_des = np.clip(-acc_des[1] / GRAVITY, -self.max_tilt, self.max_tilt)
        pitch_des = np.clip(acc_des[0] / GRAVITY, -self.max_tilt, self.max_tilt)
        att_des = np.array([roll_des, pitch_des, yaw_target])

        # ---- Attitude loop (P) -> desired body rate ----
        att_cur = quat_to_euler(quat)
        att_err = att_des - att_cur
        att_err[2] = np.arctan2(np.sin(att_err[2]), np.cos(att_err[2]))  # wrap yaw error
        rate_des = self.kp_att * att_err

        # ---- Rate loop (PID) -> desired body torque ----
        rate_err = rate_des - gyro
        self._rate_i += rate_err * self.dt
        self._rate_i = np.clip(self._rate_i, -5.0, 5.0)  # anti-windup, same reasoning
        rate_d = (rate_err - self._prev_rate_err) / self.dt
        self._prev_rate_err = rate_err
        torque_des = self.kp_rate * rate_err + self.ki_rate * self._rate_i + self.kd_rate * rate_d

        # ---- Motor mixer: exact geometric allocation, not heuristic ----
        wrench = np.array([thrust_des, torque_des[0], torque_des[1], torque_des[2]])
        ctrl = _ALLOC_INV @ wrench
        return np.clip(ctrl, 0.0, self.ctrl_limit)

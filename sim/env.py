"""
Gymnasium environment for Section 11.5's RL training loop.

One episode = one main-drone mission (x1,y1) -> (x2,y2). The attacker drone may or may
not execute one of the 4 attacks at a randomized onset, matching the onset-time labeling
already required for the LOAO / detection-latency evaluations in Section 5/8.

The RL "agent" is the detector's classification decision at each timestep (Section 11.5):
observation = physics-residual feature vector, action = predicted label in
{0:normal, 1:gps_spoof, 2:imu_corrupt, 3:actuator_injection, 4:comm_jam},
reward = the structure specified in Section 11.5 (reward near-onset detection,
penalize misses and false positives, bonus for correctly flagging an unseen attack
as anomalous under LOAO).
"""
import os
from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from .controller import ArduCopterStyleController, rotate_body_to_world, GRAVITY
from .attacker import AttackerPolicy
from .attacks import (
    AttackState, apply_gps_spoof, apply_imu_corrupt,
    apply_actuator_injection, apply_comm_jam, ATTACK_NAMES,
)
from .comm_link import CommLink, inter_drone_range
from .detector import featurize, N_FEATURES, N_CLASSES

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "battlefield_scene.xml"
)

# reward structure, Section 11.5
REWARD_CORRECT_ATTACK = 3.0        # correctly flagging an active attack
REWARD_CORRECT_NORMAL = 0.5        # correctly saying "normal" during normal flight
PENALTY_MISS = -3.0                # attack active, predicted normal (or wrong attack)
PENALTY_FALSE_POSITIVE = -1.5      # normal flight, predicted an attack
ONSET_BONUS_WINDOW = 1.0           # seconds; extra reward for catching it fast
ONSET_BONUS_MAX = 2.0
LOAO_UNSEEN_BONUS = 1.5            # extra reward for flagging-as-anomalous an attack the
                                    # detector was never trained on (Section 5 LOAO tie-in)


class TwoDroneIDSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, x1y1=(-6.0, -16.0), x2y2=(-6.0, 16.0), altitude=3.0,
                 max_episode_seconds=30.0, leave_one_out_id=None, seed=None,
                 physics_fusion_gain=0.003, attacker_standoff=12.0, attacker_side=1.0):
        super().__init__()
        self.physics_fusion_gain = physics_fusion_gain
        self.attacker_standoff = attacker_standoff
        self.attacker_side = attacker_side
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.max_steps = int(max_episode_seconds / self.dt)

        self.start_xy = np.array(x1y1, dtype=float)
        self.goal_xy = np.array(x2y2, dtype=float)
        self.altitude = altitude

        self.main_ctrl = ArduCopterStyleController(dt=self.dt)
        self.attacker = AttackerPolicy(dt=self.dt, rng=np.random.default_rng(seed))
        self.main_attack_state = AttackState()
        self.comm_link = CommLink(base_delay_steps=1)

        # LOAO support (Section 5/8): optionally exclude one attack id from ever being
        # scheduled during training, so it can be used purely as an open-set test.
        self.leave_one_out_id = leave_one_out_id

        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(N_FEATURES,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_CLASSES)

        self.rng = np.random.default_rng(seed)
        self._window = None
        self._log = None  # populated when record=True

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None, record=False, force_attack_id=None, onset_frac=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.main_ctrl.reset()
        self.attacker.reset()
        self.main_attack_state.clear()
        self.comm_link.reset()

        # decide this episode's attack (Section 11.5 curriculum: random attack, random onset)
        candidates = [1, 2, 3, 4]
        if self.leave_one_out_id is not None and self.leave_one_out_id in candidates:
            candidates.remove(self.leave_one_out_id)
        if force_attack_id is not None:
            attack_id = force_attack_id
        else:
            attack_id = int(self.rng.choice([0] + candidates))  # 0 = no attack this episode
        onset_frac = onset_frac if onset_frac is not None else float(self.rng.uniform(0.3, 0.7))
        self.attacker.schedule(attack_id, onset_frac=onset_frac,
                                start_xy=self.start_xy, goal_xy=self.goal_xy, altitude=self.altitude,
                                standoff_distance=self.attacker_standoff, side=self.attacker_side)
        self._episode_attack_id = attack_id

        mujoco.mj_resetData(self.model, self.data)
        main_start = np.array([self.start_xy[0], self.start_xy[1], self.altitude])
        # start the attacker already near its station-keeping point (a real attacker would
        # already be positioned along the observed/expected flight corridor before launch)
        att_start = (self.attacker._intercept_target
                     if self.attacker._intercept_target is not None
                     else main_start + np.array([2.5, 2.5, 0.0]))
        self.data.qpos[0:3] = main_start
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:10] = att_start
        self.data.qpos[10:14] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        trip_dist = float(np.linalg.norm(self.goal_xy - self.start_xy))
        expected_steps = max(1, int(1.4 * trip_dist / self.main_ctrl.max_speed / self.dt))
        self._expected_steps = min(expected_steps, self.max_steps)

        self._t = 0.0
        self._step_i = 0
        self._last_ctrl_cmd = np.full(4, self.main_ctrl.ctrl_limit / 4)
        self._last_ctrl_applied = self._last_ctrl_cmd.copy()
        # physics branch state: an INDEPENDENT position/velocity estimate obtained by
        # dead-reckoning the accelerometer, not by differencing the (possibly spoofed)
        # reported GPS against itself -- see the Section 11 write-up on why the first
        # version of this was a no-op (predicted == reported, always, by construction).
        self._imu_pos_est = main_start.copy()
        self._imu_vel_est = np.zeros(3)
        self._prev_reported_pos = main_start.copy()  # still used as the controller's own vel estimate

        self._window = {k: deque(maxlen=100) for k in [  # 1.0s of history (was 0.2s)
            "gps_reported", "gps_physics_pred", "gyro", "accel", "ctrl_cmd",
            "ctrl_applied", "comm_gap_steps", "comm_dropped", "mag", "vel", "path_dev",
        ]}
        self._record = record
        self._log = [] if record else None

        self._advance_physics()  # populate window with one real tick before first obs
        obs = featurize(self._window)
        info = {"true_label": 0, "attack_id": attack_id}
        return obs.astype(np.float32), info

    # ------------------------------------------------------------------ #
    def step(self, action):
        true_label = self.main_attack_state.active_id
        severity = self.main_attack_state.severity
        onset_dt = self.main_attack_state.time_since_onset(self._t)

        reward = self._reward(action, true_label, onset_dt)

        self._advance_physics()
        self._step_i += 1

        main_pos = self.data.xpos[self.model.body("main").id].copy()
        terminated = bool(np.linalg.norm(main_pos[:2] - self.goal_xy) < 0.3)
        crashed = bool(main_pos[2] > 40.0 or np.linalg.norm(main_pos[:2]) > 90.0)
        terminated = terminated or crashed
        truncated = self._step_i >= self.max_steps

        obs = featurize(self._window)
        info = {
            "true_label": true_label, "attack_id": self._episode_attack_id,
            "main_pos": main_pos.tolist(),
        }
        return obs.astype(np.float32), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _reward(self, action, true_label, onset_dt):
        if true_label == 0:
            return REWARD_CORRECT_NORMAL if action == 0 else PENALTY_FALSE_POSITIVE

        # an active attack: is the LOAO-excluded one being probed as "anomalous"?
        if self.leave_one_out_id is not None and true_label == self.leave_one_out_id:
            # detector was never trained on this class; reward it for flagging *anything*
            # other than "normal" (an open-set anomaly flag), per Section 5's LOAO protocol
            return LOAO_UNSEEN_BONUS if action != 0 else PENALTY_MISS

        if action != true_label:
            return PENALTY_MISS

        bonus = ONSET_BONUS_MAX * max(0.0, 1.0 - onset_dt / ONSET_BONUS_WINDOW)
        return REWARD_CORRECT_ATTACK + bonus

    # ------------------------------------------------------------------ #
    def _advance_physics(self):
        main_id, att_id = self.model.body("main").id, self.model.body("att").id
        main_pos = self.data.xpos[main_id].copy()
        main_vel = self.data.cvel[main_id][3:6].copy()
        main_quat = self.data.sensor("main_att_quat").data.copy()
        main_gyro = self.data.sensor("main_gyro").data.copy()
        main_accel = self.data.sensor("main_accel").data.copy()
        main_mag = self.data.sensor("main_mag").data.copy()
        main_gps_true = self.data.sensor("main_gps_pos").data.copy()

        att_pos = self.data.xpos[att_id].copy()
        att_vel = self.data.cvel[att_id][3:6].copy()
        att_quat = self.data.sensor("att_att_quat").data.copy()
        att_gyro_local = self.data.cvel[att_id][0:3].copy()

        rng_m = inter_drone_range(main_gps_true, att_pos)

        # ---- attacker: fly + maybe trigger its scheduled attack ----
        att_ctrl, _ = self.attacker.step(
            att_pos, att_vel, att_quat, att_gyro_local, main_gps_true, self._t,
            episode_frac=min(1.0, self._step_i / self._expected_steps),
            main_vel=main_vel,
        )
        self.main_attack_state = self.attacker.attack_state
        active = self.main_attack_state.active_id

        # ---- Attack 1: GPS spoof (corrupts what the main controller believes its position is) ----
        gps_reported = main_gps_true.copy()
        if active == 1:
            gps_reported = apply_gps_spoof(main_gps_true, self.main_attack_state, self._t)

        # ---- Attack 2: IMU corruption ----
        gyro_reported, accel_reported = main_gyro.copy(), main_accel.copy()
        if active == 2:
            gyro_reported, accel_reported = apply_imu_corrupt(main_gyro, main_accel, self.main_attack_state, self._t)

        # velocity estimate for the controller: finite-difference of *reported* GPS
        # (this is what the flight controller itself uses -- realistically, a spoofed
        # GPS should mislead the controller, which is the point of Attack 1)
        vel_est = (gps_reported - self._prev_reported_pos) / self.dt
        self._prev_reported_pos = gps_reported.copy()

        # ---- physics branch: independent accelerometer dead-reckoning (Section 5.3a) ----
        # deliberately does NOT touch gps_reported -- it predicts where the drone should
        # be from inertial data alone, so a GPS spoof shows up as a residual against a
        # source the spoof never corrupted. (imu corruption, in turn, corrupts the
        # accelerometer this reads, showing up as physics-branch divergence for Attack 2.)
        world_accel = rotate_body_to_world(main_quat, accel_reported) - np.array([0, 0, GRAVITY])
        gps_physics_pred = self._imu_pos_est + self._imu_vel_est * self.dt
        self._imu_vel_est = self._imu_vel_est + world_accel * self.dt
        # slow complementary correction toward GPS so IMU drift doesn't grow unbounded
        # over a full mission -- low gain on purpose, so a spoof still shows up as a
        # sustained residual rather than being absorbed away within one tick
        fusion_gain = self.physics_fusion_gain
        self._imu_pos_est = gps_physics_pred + fusion_gain * (gps_reported - gps_physics_pred)

        main_target = np.array([self.goal_xy[0], self.goal_xy[1], self.altitude])
        ctrl_cmd = self.main_ctrl.update(gps_reported, vel_est, main_quat, gyro_reported, main_target)

        # ---- Attack 3: actuator/command injection ----
        ctrl_after_attack = ctrl_cmd.copy()
        if active == 3:
            ctrl_after_attack = apply_actuator_injection(ctrl_cmd, self.main_attack_state, self._t)

        # ---- Attack 4: comm jamming -- delay/drop the control packet before it reaches the motors ----
        extra_latency, drop_prob = (0, 0.0)
        if active == 4:
            extra_latency, drop_prob = apply_comm_jam(self.main_attack_state, self._t)
        arrived = self.comm_link.push_and_pop(ctrl_after_attack, extra_latency, drop_prob, rng=self.rng)
        dropped = arrived is None
        if arrived is None:
            arrived = self._last_ctrl_applied  # open-loop hold on the last good command
        ctrl_applied = arrived

        self.data.ctrl[0:4] = ctrl_applied
        self.data.ctrl[4:8] = att_ctrl
        mujoco.mj_step(self.model, self.data)
        self._t += self.dt

        self._last_ctrl_cmd, self._last_ctrl_applied = ctrl_cmd, ctrl_applied

        w = self._window
        w["gps_reported"].append(gps_reported)
        w["gps_physics_pred"].append(gps_physics_pred)
        w["gyro"].append(gyro_reported)
        w["accel"].append(accel_reported)
        w["ctrl_cmd"].append(ctrl_cmd)
        w["ctrl_applied"].append(ctrl_applied)
        w["comm_gap_steps"].append(float(extra_latency))
        w["comm_dropped"].append(1.0 if dropped else 0.0)
        w["mag"].append(main_mag)
        w["vel"].append(vel_est)

        # cross-track deviation of the physics branch's OWN independent position estimate
        # (IMU dead-reckoning, self._imu_pos_est -- not the possibly-spoofed GPS reading)
        # from the known straight-line mission path. Using gps_reported here instead
        # would be a mistake: under GPS spoofing, the controller actively steers the
        # *true* trajectory so that the *spoofed* reading looks on-path, so cross-track
        # deviation of gps_reported stays near zero by construction, telling us nothing.
        # The dead-reckoned estimate isn't fooled by the spoof, so it's the one that
        # actually reveals the drone drifting off its real intended path.
        path_vec = self.goal_xy - self.start_xy
        path_len = np.linalg.norm(path_vec) + 1e-6
        path_dir = path_vec / path_len
        rel = self._imu_pos_est[:2] - self.start_xy
        along = np.dot(rel, path_dir)
        cross_track = rel - along * path_dir
        w["path_dev"].append(float(np.linalg.norm(cross_track)))

        if self._record:
            self._log.append({
                "t": self._t,
                "main_pos_true": main_pos.tolist(),
                "main_pos_reported": gps_reported.tolist(),
                "att_pos": att_pos.tolist(),
                "true_label": int(active),
                "range_m": rng_m,
            })

    def get_log(self):
        return self._log

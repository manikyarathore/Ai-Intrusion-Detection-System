"""
Four-drone environment (Section 11 redesign, per the two uploaded papers):

  Team A: a1 = attacker (embedded), a2 = escort (formation-flies with a1, cosmetic)
  Team B: b1 = victim (flies the patrol mission), b2 = friend (flies wingman formation
          with b1, and is the one actually running the detector + Bayesian decision)

a1 attacks b1 exactly as in env.py's two-drone case (same 4 attacks, same physics). b2
does NOT get privileged/omniscient information about b1 -- it runs the same trained
PRHN-lite detector against b1's *reported* telemetry (received over a trusted
inter-drone link, i.e. the same channel a real formation would use for state-sharing;
this is a simplifying assumption, not modeled as its own lossy/attackable channel here).
b2's classifications feed a FriendMonitor (Bayesian two-game decision, bayesian_game.py)
which decides whether to eject b1 -- requiring the paper's "persistent misbehavior"
before acting, not a single flagged tick. An ejection decision is proposed as a
blockchain block (blockchain_ledger.py) and needs quorum sign-off from a2 (an
independent third party) before it commits -- b2, as the accuser, also signs, but b1
(the subject) cannot.

Every one of these steps runs through mujoco.mj_step() -- there is no offline/background
computation standing in for the physics. See sim/train_live_four.py for training this
live inside MuJoCo's viewer, which is now the primary/recommended training path.
"""
import os
from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from .controller import ArduCopterStyleController
from .attacker import AttackerPolicy
from .attacks import (
    AttackState, apply_gps_spoof, apply_imu_corrupt,
    apply_actuator_injection, apply_comm_jam, ATTACK_NAMES,
)
from .comm_link import CommLink, inter_drone_range
from .detector import featurize, N_FEATURES, N_CLASSES, PRHNLiteDetector
from .bayesian_game import GameParams, FriendMonitor
from .blockchain_ledger import DroneBlockchain

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "four_drone_battlefield.xml")

REWARD_CORRECT_ATTACK = 3.0
REWARD_CORRECT_NORMAL = 0.5
PENALTY_MISS = -3.0
PENALTY_FALSE_POSITIVE = -1.5
ONSET_BONUS_WINDOW = 1.0
ONSET_BONUS_MAX = 2.0


class FourDroneIDSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, b1_start=(-6.0, -16.0), b1_goal=(-6.0, 16.0), altitude=3.0,
                 max_episode_seconds=30.0, seed=None, physics_fusion_gain=0.003,
                 attacker_standoff=12.0, attacker_side=1.0,
                 wingman_offset=(3.0, 0.0, 0.0), escort_offset=(0.0, 6.0, 0.0)):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.max_steps = int(max_episode_seconds / self.dt)
        self.physics_fusion_gain = physics_fusion_gain

        self.start_xy = np.array(b1_start, dtype=float)
        self.goal_xy = np.array(b1_goal, dtype=float)
        self.altitude = altitude
        self.wingman_offset = np.array(wingman_offset)
        self.escort_offset = np.array(escort_offset)
        self.attacker_standoff = attacker_standoff
        self.attacker_side = attacker_side

        # controllers: b1 (mission), b2 (wingman formation), a2 (escort formation)
        self.b1_ctrl = ArduCopterStyleController(dt=self.dt)
        self.b2_ctrl = ArduCopterStyleController(dt=self.dt, max_speed=2.6)
        self.a2_ctrl = ArduCopterStyleController(dt=self.dt, max_speed=2.6)
        self.attacker = AttackerPolicy(dt=self.dt, rng=np.random.default_rng(seed))
        self.b1_attack_state = AttackState()
        self.comm_link = CommLink(base_delay_steps=1)

        # actuator index ranges, looked up by name (robust to XML ordering changes)
        self._act_idx = {
            n: [self.model.actuator(f"{n}_thrust{i}").id for i in range(1, 5)]
            for n in ("a1", "a2", "b1", "b2")
        }

        # b2's onboard detector + Bayesian friend-monitor + shared fleet ledger
        self.detector = PRHNLiteDetector(seed=seed or 0)
        self._detector_loaded = False
        self.friend_monitor = FriendMonitor(params=GameParams())
        self.blockchain = DroneBlockchain(drone_ids=["a1", "a2", "b1", "b2"], quorum=2,
                                           seed=seed or 0)
        self._ejection_logged = False

        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(N_FEATURES,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_CLASSES)
        self.rng = np.random.default_rng(seed)

    def load_detector(self, path):
        self.detector.load(path)
        self._detector_loaded = True

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None, record=False, force_attack_id=None, onset_frac=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        candidates = [1, 2, 3, 4]
        attack_id = force_attack_id if force_attack_id is not None else int(self.rng.choice([0] + candidates))
        onset_frac = onset_frac if onset_frac is not None else float(self.rng.uniform(0.3, 0.7))
        self.attacker.schedule(attack_id, onset_frac=onset_frac, start_xy=self.start_xy,
                                goal_xy=self.goal_xy, altitude=self.altitude,
                                standoff_distance=self.attacker_standoff, side=self.attacker_side)
        self._episode_attack_id = attack_id

        b1_start = np.array([self.start_xy[0], self.start_xy[1], self.altitude])
        b2_start = b1_start + self.wingman_offset
        a1_start = (self.attacker._intercept_target if self.attacker._intercept_target is not None
                    else b1_start + np.array([14.0, 0.0, 0.0]))
        a2_start = a1_start + self.escort_offset

        for name, pos in [("a1", a1_start), ("a2", a2_start), ("b1", b1_start), ("b2", b2_start)]:
            qadr = self.model.body(name).jntadr[0]
            qpos_adr = self.model.jnt_qposadr[qadr]
            self.data.qpos[qpos_adr:qpos_adr + 3] = pos
            self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        self.b1_ctrl.reset(); self.b2_ctrl.reset(); self.a2_ctrl.reset()
        self.attacker.reset()
        self.b1_attack_state.clear()
        self.comm_link.reset()
        self.friend_monitor = FriendMonitor(params=GameParams())
        self._ejection_logged = False

        trip_dist = float(np.linalg.norm(self.goal_xy - self.start_xy))
        expected_steps = max(1, int(1.4 * trip_dist / self.b1_ctrl.max_speed / self.dt))
        self._expected_steps = min(expected_steps, self.max_steps)

        self._t = 0.0
        self._step_i = 0
        self._last_ctrl_cmd = np.full(4, self.b1_ctrl.ctrl_limit / 4)
        self._last_ctrl_applied = self._last_ctrl_cmd.copy()
        self._imu_pos_est = b1_start.copy()
        self._imu_vel_est = np.zeros(3)
        self._prev_reported_pos = b1_start.copy()

        self._window = {k: deque(maxlen=100) for k in [
            "gps_reported", "gps_physics_pred", "gyro", "accel", "ctrl_cmd",
            "ctrl_applied", "comm_gap_steps", "comm_dropped", "mag", "vel", "path_dev",
        ]}
        self._record = record
        self._log = [] if record else None

        self._advance_physics()
        obs = featurize(self._window)
        info = {"true_label": 0, "attack_id": attack_id}
        return obs.astype(np.float32), info

    # ------------------------------------------------------------------ #
    def step(self, action):
        true_label = self.b1_attack_state.active_id
        onset_dt = self.b1_attack_state.time_since_onset(self._t)
        reward = self._reward(action, true_label, onset_dt)

        self._advance_physics()
        self._step_i += 1

        # ---- Bayesian friend-monitor + blockchain (deterministic, not RL-trained) ----
        friend_status = self.friend_monitor.update(detector_flagged_attack=(action != 0))
        if friend_status["ejected"] and not self._ejection_logged:
            self._ejection_logged = True
            block = self.blockchain.propose_block({
                "type": "ejection_decision", "subject_drone": "b1", "proposer": "b2",
                "mr": friend_status["mr"], "decision": "EJECT", "t": round(self._t, 2),
            })
            self.blockchain.sign_pending("b2")   # accuser signs
            self.blockchain.sign_pending("a2")   # independent third-party witness signs

        b1_pos = self.data.xpos[self.model.body("b1").id].copy()
        terminated = bool(np.linalg.norm(b1_pos[:2] - self.goal_xy) < 0.3)
        crashed = bool(b1_pos[2] > 40.0 or np.linalg.norm(b1_pos[:2]) > 90.0)
        terminated = terminated or crashed
        truncated = self._step_i >= self.max_steps

        obs = featurize(self._window)
        info = {
            "true_label": true_label, "attack_id": self._episode_attack_id,
            "main_pos": b1_pos.tolist(), "friend_status": friend_status,
            "blockchain_len": len(self.blockchain),
        }
        return obs.astype(np.float32), reward, terminated, truncated, info

    def _reward(self, action, true_label, onset_dt):
        if true_label == 0:
            return REWARD_CORRECT_NORMAL if action == 0 else PENALTY_FALSE_POSITIVE
        if action != true_label:
            return PENALTY_MISS
        bonus = ONSET_BONUS_MAX * max(0.0, 1.0 - onset_dt / ONSET_BONUS_WINDOW)
        return REWARD_CORRECT_ATTACK + bonus

    # ------------------------------------------------------------------ #
    def _advance_physics(self):
        b1_id, a1_id, a2_id, b2_id = (self.model.body(n).id for n in ("b1", "a1", "a2", "b2"))
        b1_pos = self.data.xpos[b1_id].copy()
        b1_vel = self.data.cvel[b1_id][3:6].copy()
        b1_quat = self.data.sensor("b1_att_quat").data.copy()
        b1_gyro = self.data.sensor("b1_gyro").data.copy()
        b1_accel = self.data.sensor("b1_accel").data.copy()
        b1_mag = self.data.sensor("b1_mag").data.copy()
        b1_gps_true = self.data.sensor("b1_gps_pos").data.copy()

        a1_pos = self.data.xpos[a1_id].copy()
        a1_vel = self.data.cvel[a1_id][3:6].copy()
        a1_quat = self.data.sensor("a1_att_quat").data.copy()
        a1_gyro_local = self.data.cvel[a1_id][0:3].copy()

        att_ctrl, _ = self.attacker.step(
            a1_pos, a1_vel, a1_quat, a1_gyro_local, b1_gps_true, self._t,
            episode_frac=min(1.0, self._step_i / self._expected_steps),
        )
        self.b1_attack_state = self.attacker.attack_state
        active = self.b1_attack_state.active_id

        gps_reported = b1_gps_true.copy()
        if active == 1:
            gps_reported = apply_gps_spoof(b1_gps_true, self.b1_attack_state, self._t)
        gyro_reported, accel_reported = b1_gyro.copy(), b1_accel.copy()
        if active == 2:
            gyro_reported, accel_reported = apply_imu_corrupt(b1_gyro, b1_accel, self.b1_attack_state, self._t)

        vel_est = (gps_reported - self._prev_reported_pos) / self.dt
        self._prev_reported_pos = gps_reported.copy()

        from .controller import rotate_body_to_world, GRAVITY
        world_accel = rotate_body_to_world(b1_quat, accel_reported) - np.array([0, 0, GRAVITY])
        gps_physics_pred = self._imu_pos_est + self._imu_vel_est * self.dt
        self._imu_vel_est = self._imu_vel_est + world_accel * self.dt
        fusion_gain = self.physics_fusion_gain
        self._imu_pos_est = gps_physics_pred + fusion_gain * (gps_reported - gps_physics_pred)

        b1_target = np.array([self.goal_xy[0], self.goal_xy[1], self.altitude])
        ctrl_cmd = self.b1_ctrl.update(gps_reported, vel_est, b1_quat, gyro_reported, b1_target)

        ctrl_after_attack = ctrl_cmd.copy()
        if active == 3:
            ctrl_after_attack = apply_actuator_injection(ctrl_cmd, self.b1_attack_state, self._t)
        extra_latency, drop_prob = (0, 0.0)
        if active == 4:
            extra_latency, drop_prob = apply_comm_jam(self.b1_attack_state, self._t)
        arrived = self.comm_link.push_and_pop(ctrl_after_attack, extra_latency, drop_prob, rng=self.rng)
        dropped = arrived is None
        ctrl_applied = arrived if arrived is not None else self._last_ctrl_applied

        # ---- formation flight for a2 (escort) and b2 (wingman) -- smooth, station-keeping,
        # constant-altitude follow, so the pairs visually read as flying together ----
        a2_pos = self.data.xpos[a2_id].copy()
        a2_vel = self.data.cvel[a2_id][3:6].copy()
        a2_quat = self.data.sensor("a2_att_quat").data.copy()
        a2_gyro = self.data.cvel[a2_id][0:3].copy()
        a2_target = a1_pos + self.escort_offset
        a2_ctrl = self.a2_ctrl.update(a2_pos, a2_vel, a2_quat, a2_gyro, a2_target)

        b2_pos = self.data.xpos[b2_id].copy()
        b2_vel = self.data.cvel[b2_id][3:6].copy()
        b2_quat = self.data.sensor("b2_att_quat").data.copy()
        b2_gyro = self.data.cvel[b2_id][0:3].copy()
        b2_target = b1_pos + self.wingman_offset
        b2_ctrl = self.b2_ctrl.update(b2_pos, b2_vel, b2_quat, b2_gyro, b2_target)

        self.data.ctrl[self._act_idx["a1"]] = att_ctrl
        self.data.ctrl[self._act_idx["a2"]] = a2_ctrl
        self.data.ctrl[self._act_idx["b1"]] = ctrl_applied
        self.data.ctrl[self._act_idx["b2"]] = b2_ctrl
        mujoco.mj_step(self.model, self.data)
        self._t += self.dt

        self._last_ctrl_cmd, self._last_ctrl_applied = ctrl_cmd, ctrl_applied

        path_vec = self.goal_xy - self.start_xy
        path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
        rel = self._imu_pos_est[:2] - self.start_xy
        along = np.dot(rel, path_dir)
        cross_track = rel - along * path_dir

        w = self._window
        w["gps_reported"].append(gps_reported)
        w["gps_physics_pred"].append(gps_physics_pred)
        w["gyro"].append(gyro_reported)
        w["accel"].append(accel_reported)
        w["ctrl_cmd"].append(ctrl_cmd)
        w["ctrl_applied"].append(ctrl_applied)
        w["comm_gap_steps"].append(float(extra_latency))
        w["comm_dropped"].append(1.0 if dropped else 0.0)
        w["mag"].append(b1_mag)
        w["vel"].append(vel_est)
        w["path_dev"].append(float(np.linalg.norm(cross_track)))

        if self._record:
            self._log.append({
                "t": round(self._t, 3),
                "a1": a1_pos.tolist(), "a2": a2_pos.tolist(),
                "b1": b1_pos.tolist(), "b2": b2_pos.tolist(),
                "true_label": int(active),
            })

    def get_log(self):
        return self._log

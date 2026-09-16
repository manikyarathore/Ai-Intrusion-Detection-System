"""
2v2 version of sim/ardupilot_bridge.py: real ArduCopter SITL firmware flies b1 through
this MuJoCo physics, while a1 (attacker), a2 (escort), and b2 (friend/monitor) fly with
the in-sim ArduCopter-*style* controller as usual. b2 runs the same live detection +
Bayesian ejection decision + blockchain logging as env_four.py, so swapping b1's flight
control for real firmware doesn't change anything about how it gets monitored or judged
by its teammate.

Same honesty note as sim/ardupilot_bridge.py: this bridge's socket/JSON mechanics are
implemented against ArduPilot's publicly documented JSON FDM protocol and are internally
testable (see the loopback tests run during development), but this has **not** been
verified end-to-end against a real ArduCopter SITL binary, since compiling that firmware
wasn't feasible in the sandboxed environment this project was built in.

Improvement over the original 1v1 bridge: that version loaded a detector but never
actually called it during the bridge loop, so it never wired up live detection at all --
here, b2's detector genuinely runs every tick, exactly as it would with the normal
FourDroneIDSEnv.

Setup: same as sim/ardupilot_bridge.py's module docstring -- install & build ArduPilot
on your own machine, point ArduCopter SITL's JSON backend at this bridge's port, then:
    python -m sim.ardupilot_bridge_four --attack 1
"""
import argparse
import json
import socket
import time
from collections import deque

import numpy as np
import mujoco

from .env_four import MODEL_PATH
from .controller import ArduCopterStyleController, rotate_body_to_world, GRAVITY
from .attacker import AttackerPolicy
from .attacks import (
    AttackState, apply_gps_spoof, apply_imu_corrupt,
    apply_actuator_injection, apply_comm_jam,
)
from .comm_link import CommLink
from .detector import featurize, PRHNLiteDetector
from .bayesian_game import GameParams, FriendMonitor
from .blockchain_ledger import DroneBlockchain

PWM_MIN, PWM_MAX = 1000.0, 2000.0
CTRL_MIN, CTRL_MAX = 0.0, 13.0


def pwm_to_ctrl(pwm):
    frac = np.clip((np.asarray(pwm) - PWM_MIN) / (PWM_MAX - PWM_MIN), 0.0, 1.0)
    return frac * CTRL_MAX


class ArduPilotJSONBridge:
    def __init__(self, listen_port=9002, send_port=9003, send_addr="127.0.0.1"):
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_sock.bind(("0.0.0.0", listen_port))
        self.listen_sock.settimeout(0.05)
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (send_addr, send_port)
        print(f"ArduPilot JSON bridge listening on :{listen_port}, replying to {send_addr}:{send_port}")

    def recv_pwm(self):
        try:
            data, _ = self.listen_sock.recvfrom(4096)
            packet = json.loads(data.decode("utf-8"))
            return packet.get("pwm")
        except (socket.timeout, json.JSONDecodeError):
            return None

    def send_state(self, t, gyro, accel_body, pos, quat, vel):
        packet = {
            "timestamp": t,
            "imu": {"gyro": list(gyro), "accel_body": list(accel_body)},
            "position": list(pos), "quaternion": list(quat), "velocity": list(vel),
        }
        self.send_sock.sendto(json.dumps(packet).encode("utf-8"), self.send_addr)


def run_bridge(attack_id=1, onset_frac=0.4, max_episode_seconds=30.0, seed=7,
               detector_path="runs/detector_four.npz", listen_port=9002, send_port=9003,
               physics_fusion_gain=0.003):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    bridge = ArduPilotJSONBridge(listen_port=listen_port, send_port=send_port)
    a2_ctrl = ArduCopterStyleController(dt=dt, max_speed=2.6)
    b2_ctrl = ArduCopterStyleController(dt=dt, max_speed=2.6)
    attacker = AttackerPolicy(dt=dt, rng=np.random.default_rng(seed))
    comm_link = CommLink(base_delay_steps=1)

    detector = PRHNLiteDetector(seed=seed)
    try:
        detector.load(detector_path)
        print(f"loaded trained detector from {detector_path}")
    except Exception as e:
        print(f"no trained detector ({e}); b2 will run with an untrained detector")

    friend_monitor = FriendMonitor(params=GameParams())
    blockchain = DroneBlockchain(drone_ids=["a1", "a2", "b1", "b2"], quorum=2, seed=seed)
    ejection_logged = False

    start_xy = np.array([-6.0, -16.0])
    goal_xy = np.array([-6.0, 16.0])
    altitude = 3.0
    wingman_offset = np.array([3.0, 0.0, 0.0])
    escort_offset = np.array([0.0, 6.0, 0.0])

    attacker.schedule(attack_id, onset_frac=onset_frac, start_xy=start_xy, goal_xy=goal_xy,
                       altitude=altitude, standoff_distance=12.0, side=1.0)

    b1_start = np.array([start_xy[0], start_xy[1], altitude])
    b2_start = b1_start + wingman_offset
    a1_start = attacker._intercept_target
    a2_start = a1_start + escort_offset
    for name, pos in [("a1", a1_start), ("a2", a2_start), ("b1", b1_start), ("b2", b2_start)]:
        qadr = model.jnt_qposadr[model.body(name).jntadr[0]]
        data.qpos[qadr:qadr + 3] = pos
        data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # physics-branch dead-reckoning state for b2's view of b1 (same as env_four.py)
    imu_pos_est = b1_start.copy()
    imu_vel_est = np.zeros(3)
    prev_reported_pos = b1_start.copy()
    window = {k: deque(maxlen=100) for k in [
        "gps_reported", "gps_physics_pred", "gyro", "accel", "ctrl_cmd",
        "ctrl_applied", "comm_gap_steps", "comm_dropped", "mag", "vel", "path_dev",
    ]}

    print("Waiting for ArduCopter SITL to connect and start sending PWM packets for b1...")
    last_ctrl = np.full(4, 3.25)
    t = 0.0
    max_steps = int(max_episode_seconds / dt)

    for step_i in range(max_steps):
        pwm = bridge.recv_pwm()
        if pwm is not None:
            last_ctrl = pwm_to_ctrl(pwm[:4])

        b1_gyro = data.sensor("b1_gyro").data.copy()
        b1_accel = data.sensor("b1_accel").data.copy()
        b1_mag = data.sensor("b1_mag").data.copy()
        b1_quat = data.sensor("b1_att_quat").data.copy()
        b1_gps_true = data.sensor("b1_gps_pos").data.copy()
        b1_vel = data.cvel[model.body("b1").id][3:6].copy()

        a1_pos = data.xpos[model.body("a1").id].copy()
        a1_vel = data.cvel[model.body("a1").id][3:6].copy()
        a1_quat = data.sensor("a1_att_quat").data.copy()
        a1_gyro = data.cvel[model.body("a1").id][0:3].copy()
        att_ctrl, _ = attacker.step(a1_pos, a1_vel, a1_quat, a1_gyro, b1_gps_true, t,
                                     episode_frac=min(1.0, step_i / (max_steps * 0.6)))
        active = attacker.attack_state.active_id

        gyro_reported, accel_reported = b1_gyro, b1_accel
        if active == 2:
            gyro_reported, accel_reported = apply_imu_corrupt(b1_gyro, b1_accel, attacker.attack_state, t)

        ctrl_cmd = last_ctrl.copy()  # what ArduCopter SITL commanded, pre-attack
        ctrl_to_apply = ctrl_cmd.copy()
        if active == 3:
            ctrl_to_apply = apply_actuator_injection(ctrl_cmd, attacker.attack_state, t)
        extra_latency, drop_prob = 0, 0.0
        if active == 4:
            extra_latency, drop_prob = apply_comm_jam(attacker.attack_state, t)
            arrived = comm_link.push_and_pop(ctrl_to_apply, extra_latency, drop_prob)
            ctrl_to_apply = arrived if arrived is not None else last_ctrl
        dropped = active == 4 and extra_latency > 0

        gps_reported = b1_gps_true.copy()
        if active == 1:
            gps_reported = apply_gps_spoof(b1_gps_true, attacker.attack_state, t)

        vel_est = (gps_reported - prev_reported_pos) / dt
        prev_reported_pos = gps_reported.copy()
        world_accel = rotate_body_to_world(b1_quat, accel_reported) - np.array([0, 0, GRAVITY])
        gps_physics_pred = imu_pos_est + imu_vel_est * dt
        imu_vel_est = imu_vel_est + world_accel * dt
        imu_pos_est = gps_physics_pred + physics_fusion_gain * (gps_reported - gps_physics_pred)

        # ---- a2 (escort) and b2 (wingman) formation flight ----
        a2_pos = data.xpos[model.body("a2").id].copy()
        a2_vel = data.cvel[model.body("a2").id][3:6].copy()
        a2_quat = data.sensor("a2_att_quat").data.copy()
        a2_gyro = data.cvel[model.body("a2").id][0:3].copy()
        a2_ctrl_out = a2_ctrl.update(a2_pos, a2_vel, a2_quat, a2_gyro, a1_pos + escort_offset)

        b2_pos = data.xpos[model.body("b2").id].copy()
        b2_vel = data.cvel[model.body("b2").id][3:6].copy()
        b2_quat = data.sensor("b2_att_quat").data.copy()
        b2_gyro = data.cvel[model.body("b2").id][0:3].copy()
        b1_pos_true = data.xpos[model.body("b1").id].copy()
        b2_ctrl_out = b2_ctrl.update(b2_pos, b2_vel, b2_quat, b2_gyro, b1_pos_true + wingman_offset)

        data.ctrl[[model.actuator(f"a1_thrust{i}").id for i in range(1, 5)]] = att_ctrl
        data.ctrl[[model.actuator(f"a2_thrust{i}").id for i in range(1, 5)]] = a2_ctrl_out
        data.ctrl[[model.actuator(f"b1_thrust{i}").id for i in range(1, 5)]] = ctrl_to_apply
        data.ctrl[[model.actuator(f"b2_thrust{i}").id for i in range(1, 5)]] = b2_ctrl_out
        mujoco.mj_step(model, data)
        t += dt

        bridge.send_state(t, gyro_reported, accel_reported, gps_reported, b1_quat, vel_est)

        # ---- b2's live detection + Bayesian ejection + blockchain (the actual point
        # of porting this bridge -- the original 1v1 version never did this) ----
        path_vec = goal_xy - start_xy
        path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
        rel = imu_pos_est[:2] - start_xy
        cross_track = rel - np.dot(rel, path_dir) * path_dir

        window["gps_reported"].append(gps_reported); window["gps_physics_pred"].append(gps_physics_pred)
        window["gyro"].append(gyro_reported); window["accel"].append(accel_reported)
        window["ctrl_cmd"].append(ctrl_cmd); window["ctrl_applied"].append(ctrl_to_apply)
        window["comm_gap_steps"].append(float(extra_latency)); window["comm_dropped"].append(1.0 if dropped else 0.0)
        window["mag"].append(b1_mag); window["vel"].append(vel_est)
        window["path_dev"].append(float(np.linalg.norm(cross_track)))

        obs = featurize(window)
        action, proba, cache = detector.act(obs, explore=False)
        friend_status = friend_monitor.update(detector_flagged_attack=(action != 0))

        if friend_status["ejected"] and not ejection_logged:
            ejection_logged = True
            blockchain.propose_block({
                "type": "ejection_decision", "subject_drone": "b1", "proposer": "b2",
                "mr": friend_status["mr"], "decision": "EJECT", "t": round(t, 2),
            })
            blockchain.sign_pending("b2"); blockchain.sign_pending("a2")
            print(f"   >> b2 proposed + a2 co-signed an EJECT block for b1 "
                  f"(MR={friend_status['mr']:.2f}) -- ledger length {len(blockchain)}, "
                  f"valid={blockchain.is_valid()}")

        if step_i % 500 == 0:
            print(f"t={t:5.1f}s active_attack={active} b2_pred={action} "
                  f"MR={friend_status['mr']:.3f} monitoring={friend_status['monitoring_active']} "
                  f"ejected={friend_status['ejected']}")

        time.sleep(max(0.0, dt - 0.0005))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=int, default=1)
    p.add_argument("--onset", type=float, default=0.4)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--detector", type=str, default="runs/detector_four.npz")
    p.add_argument("--listen-port", type=int, default=9002)
    p.add_argument("--send-port", type=int, default=9003)
    args = p.parse_args()
    run_bridge(attack_id=args.attack, onset_frac=args.onset, max_episode_seconds=args.seconds,
               seed=args.seed, detector_path=args.detector,
               listen_port=args.listen_port, send_port=args.send_port)

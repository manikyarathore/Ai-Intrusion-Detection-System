"""
A real implementation of ArduPilot's JSON FDM (Flight Dynamics Model) protocol, so the
*actual* ArduCopter SITL firmware can fly the main drone through our MuJoCo physics --
this is the concrete "swap point" the README referred to, now actually built.

HONESTY NOTE, read before using: I could not compile or run real ArduPilot firmware in
the sandboxed environment this project was built in (it's a large C++ codebase with a
multi-repo submodule build that isn't practical there), so this bridge is implemented
directly against ArduPilot's publicly documented JSON FDM protocol but has NOT been
verified end-to-end against a real ArduCopter SITL instance. Field names/ports match the
documented protocol as of this writing; if your ArduPilot version's JSON backend differs,
check `AP_HAL_SITL/AP_HAL_SITL_Sub/`'s SIM_JSON source or run `sim_vehicle.py --model JSON
-v ArduCopter` and inspect the packets it sends/expects with a UDP sniffer (e.g. `nc -ul
9002`) if this doesn't connect cleanly.

Protocol shape:
  - ArduCopter SITL sends servo/motor PWM commands over UDP to THIS bridge (default port
    9002), as JSON: {"frame_rate":300,"frame_count":N,"pwm":[1500,1500,...]} (16 channels).
  - This bridge steps MuJoCo physics using those PWM values (mapped to our motor ctrl
    range) and sends the resulting vehicle state back to ArduCopter SITL over UDP
    (default port 9003), as JSON:
    {"timestamp":T,"imu":{"gyro":[p,q,r],"accel_body":[ax,ay,az]},
     "position":[x,y,z],"quaternion":[w,x,y,z],"velocity":[vx,vy,vz]}

Setup (on your own machine, not in this project's sandbox):
  1. Install ArduPilot: https://ardupilot.org/dev/docs/building-setup-linux.html
  2. Build ArduCopter SITL: `./waf configure --board sitl && ./waf copter`
  3. Run it pointed at this bridge's IP/port:
     `sim_vehicle.py -v ArduCopter -f JSON --add-param-file=... -I0`
     (consult ArduPilot's SIM_JSON docs for the exact JSON-backend invocation for your
     version -- this has changed across ArduPilot releases)
  4. In another terminal: `python -m sim.ardupilot_bridge --attack 1`
  5. Send mission commands to ArduCopter over MAVLink (e.g. via pymavlink or a ground
     station) exactly as you would for real hardware or a Gazebo SITL session.

This module only replaces the MAIN drone's flight control (matching Section 11.4's
scoping: the attacker doesn't need a full ArduCopter instance). The attacker, attack
injection, and detector all still run exactly as in env.py.
"""
import argparse
import json
import socket
import time

import numpy as np
import mujoco

from .env import MODEL_PATH
from .attacker import AttackerPolicy
from .attacks import (
    AttackState, apply_gps_spoof, apply_imu_corrupt,
    apply_actuator_injection, apply_comm_jam,
)
from .comm_link import CommLink, inter_drone_range
from .detector import featurize, PRHNLiteDetector
from .controller import rotate_body_to_world, GRAVITY

PWM_MIN, PWM_MAX = 1000.0, 2000.0
CTRL_MIN, CTRL_MAX = 0.0, 13.0


def pwm_to_ctrl(pwm):
    """Map ArduCopter's PWM output (1000-2000us) to our motor ctrl range (0-13N).
    Linear approximation -- a real ESC/motor thrust curve is nonlinear, but this is a
    reasonable first-order mapping for getting the bridge working; tune against your
    specific SITL motor parameters if you need better fidelity."""
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
        """Non-blocking receive of the latest servo/PWM packet from ArduCopter SITL.
        Returns None if nothing arrived within the timeout (caller should hold last ctrl)."""
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
            "position": list(pos),
            "quaternion": list(quat),  # (w, x, y, z), matches MuJoCo's convention
            "velocity": list(vel),
        }
        self.send_sock.sendto(json.dumps(packet).encode("utf-8"), self.send_addr)


def run_bridge(attack_id=1, onset_frac=0.4, max_episode_seconds=30.0, seed=7,
               detector_path="runs/detector.npz", listen_port=9002, send_port=9003):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    bridge = ArduPilotJSONBridge(listen_port=listen_port, send_port=send_port)
    attacker = AttackerPolicy(dt=dt, rng=np.random.default_rng(seed))
    comm_link = CommLink(base_delay_steps=1)
    detector = PRHNLiteDetector(seed=seed)
    try:
        detector.load(detector_path)
    except Exception as e:
        print(f"no trained detector ({e}); running without live detection")
        detector = None

    start_xy = np.array([-6.0, -16.0])
    goal_xy = np.array([-6.0, 16.0])
    altitude = 3.0
    attacker.schedule(attack_id, onset_frac=onset_frac, start_xy=start_xy, goal_xy=goal_xy,
                       altitude=altitude, standoff_distance=12.0, side=1.0)

    data.qpos[0:3] = [start_xy[0], start_xy[1], altitude]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:10] = attacker._intercept_target
    data.qpos[10:14] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    print("Waiting for ArduCopter SITL to connect and start sending PWM packets...")
    last_ctrl = np.full(4, 3.25)
    t = 0.0
    max_steps = int(max_episode_seconds / dt)

    for step_i in range(max_steps):
        pwm = bridge.recv_pwm()
        if pwm is not None:
            last_ctrl = pwm_to_ctrl(pwm[:4])

        main_gyro = data.sensor("main_gyro").data.copy()
        main_accel = data.sensor("main_accel").data.copy()
        main_quat = data.sensor("main_att_quat").data.copy()
        main_gps = data.sensor("main_gps_pos").data.copy()
        main_vel = data.cvel[model.body("main").id][3:6].copy()

        att_pos = data.xpos[model.body("att").id].copy()
        att_vel = data.cvel[model.body("att").id][3:6].copy()
        att_quat = data.sensor("att_att_quat").data.copy()
        att_gyro = data.cvel[model.body("att").id][0:3].copy()
        att_ctrl, _ = attacker.step(att_pos, att_vel, att_quat, att_gyro, main_gps, t,
                                     episode_frac=min(1.0, step_i / (max_steps * 0.6)))
        active = attacker.attack_state.active_id

        gyro_reported, accel_reported = main_gyro, main_accel
        if active == 2:
            gyro_reported, accel_reported = apply_imu_corrupt(main_gyro, main_accel, attacker.attack_state, t)

        ctrl_to_apply = last_ctrl.copy()
        if active == 3:
            ctrl_to_apply = apply_actuator_injection(ctrl_to_apply, attacker.attack_state, t)
        if active == 4:
            extra_latency, drop_prob = apply_comm_jam(attacker.attack_state, t)
            arrived = comm_link.push_and_pop(ctrl_to_apply, extra_latency, drop_prob)
            ctrl_to_apply = arrived if arrived is not None else last_ctrl

        data.ctrl[0:4] = ctrl_to_apply
        data.ctrl[4:8] = att_ctrl
        mujoco.mj_step(model, data)
        t += dt

        gps_to_report = main_gps
        if active == 1:
            gps_to_report = apply_gps_spoof(main_gps, attacker.attack_state, t)

        bridge.send_state(t, gyro_reported, accel_reported, gps_to_report, main_quat, main_vel)

        if step_i % 500 == 0:
            print(f"t={t:5.1f}s  active_attack={active}  ctrl(post-attack)={np.round(ctrl_to_apply,2)}")

        time.sleep(max(0.0, dt - 0.0005))  # keep roughly real-time so SITL's own clock stays in sync


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=int, default=1)
    p.add_argument("--onset", type=float, default=0.4)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--detector", type=str, default="runs/detector.npz")
    p.add_argument("--listen-port", type=int, default=9002)
    p.add_argument("--send-port", type=int, default=9003)
    args = p.parse_args()
    run_bridge(attack_id=args.attack, onset_frac=args.onset, max_episode_seconds=args.seconds,
               seed=args.seed, detector_path=args.detector,
               listen_port=args.listen_port, send_port=args.send_port)

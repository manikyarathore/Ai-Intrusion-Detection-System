import json
import time
import numpy as np

import mujoco
import mujoco.viewer

# ==========================================================
# Configuration
# ==========================================================

XML_FILE = "two_drone_scene.xml"
LOG_FILE = "episode.json"

# Number of interpolation steps between every recorded frame.
# Increase for smoother playback.
SUBSTEPS = 5

# ==========================================================
# Load MuJoCo model
# ==========================================================

model = mujoco.MjModel.from_xml_path(XML_FILE)
data = mujoco.MjData(model)

with open(LOG_FILE, "r") as f:
    episode = json.load(f)

frames = episode["frames"]
dt = episode.get("dt_frame", 0.02)

print("=" * 60)
print("Scenario :", episode["attack_name"])
print("Attack ID:", episode["attack_id"])
print("Frames   :", len(frames))
print("Interpolation :", SUBSTEPS, "steps")
print("=" * 60)

# Hover orientation (identity quaternion)
identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

# ==========================================================
# Viewer
# ==========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Camera
    viewer.cam.distance = 8.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.lookat[:] = [0.0, 0.0, 1.2]

    print("\nViewer Started.")
    print("Close the viewer window to stop.\n")

    playback = 1

    while viewer.is_running():

        print(f"\n========== Playback {playback} ==========\n")

        for i in range(len(frames) - 1):

            if not viewer.is_running():
                break

            f1 = frames[i]
            f2 = frames[i + 1]

            # Current and next positions
            main1 = np.array(f1["main"], dtype=float)
            main2 = np.array(f2["main"], dtype=float)

            att1 = np.array(f1["att"], dtype=float)
            att2 = np.array(f2["att"], dtype=float)

            # ------------------------------------------------
            # Interpolate between two recorded frames
            # ------------------------------------------------
            for step in range(SUBSTEPS):

                if not viewer.is_running():
                    break

                alpha = (step + 1) / SUBSTEPS

                main = (1.0 - alpha) * main1 + alpha * main2
                att = (1.0 - alpha) * att1 + alpha * att2

                # Main drone
                data.qpos[0:3] = main
                data.qpos[3:7] = identity_quat

                # Attacker drone
                data.qpos[7:10] = att
                data.qpos[10:14] = identity_quat

                # Zero velocity because this is playback
                data.qvel[:] = 0.0

                mujoco.mj_forward(model, data)

                # Camera follows midpoint
                center = (main + att) / 2.0
                viewer.cam.lookat[:] = center

                viewer.sync()

                time.sleep(dt / SUBSTEPS)

            print(
                f"t={f2['t']:6.2f} | "
                f"GT={f2['true_label']} | "
                f"Pred={f2['pred_label']} | "
                f"Range={f2['range_m']:.2f} m",
                end="\r",
                flush=True,
            )

        playback += 1

        print("\nRestarting playback...\n")

        time.sleep(0.5)

print("\nViewer Closed.")

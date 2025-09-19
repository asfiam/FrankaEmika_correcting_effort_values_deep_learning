#!/usr/bin/env python3

import os
import csv
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import Image, JointState
import cv2
from cv_bridge import CvBridge
import pandas as pd
import glob

# ----------------------------
# Config
# ----------------------------

parent_folders = ["/jazzy_ws/ros2bags_gearbox_1",
        "/jazzy_ws/ros2bags_LObject_4"        
]

mcap_files = []

ignore_files = {"/jazzy_ws/ros2bags_LObject_4/test2/test2_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test3/test3_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test4/test4_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test5/test5_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test6/test6_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test7/test7_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test8/test8_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test9/test9_0.mcap",
        "/jazzy_ws/ros2bags_LObject_4/test10/test10_0.mcap"
}

for parent_folder in parent_folders:
    mcap_files_in_folder = glob.glob(os.path.join(parent_folder, "**", "*.mcap"), recursive = True)
    mcap_files.extend(mcap_files_in_folder)

mcap_files = [
    f for f in mcap_files
    if f not in ignore_files
]

print(len(mcap_files))

# mcap_files = [
#     "/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case5/test25/test25_0.mcap",
#     "/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case2/test9/test9_0.mcap",
#     "/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case2/test10/test10_0.mcap",
#     "/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case1/test1/test1_0.mcap"
# ]

topics = {
    "wrench": [
        "/franka_effort_real",
        "/franka_effort_sensor_to_hand",
        "/franka_effort_link7_to_sensor"
    ],
    "ee_pose": ["/franka_EE_pose"],
    "joint_states": ["/joint_states"],
    "images": [
        "/floating_camera_rgb",
        "/floating_camera_depth",
        "/wrist_camera_rgb",
        "/wrist_camera_depth"
    ]
}

bridge = CvBridge()
image_folder = "rosbag_images_from_cameras"
os.makedirs(image_folder, exist_ok=True)

csv_filename = "data_from_all_rosbags.csv"

# If fresh run → remove old CSV
if os.path.exists(csv_filename):
    os.remove(csv_filename)

# ----------------------------
# Helpers
# ----------------------------
def save_image(msg, prefix, t):
    cv_img = bridge.imgmsg_to_cv2(msg)
    filename = os.path.join(image_folder, f"{prefix}_{t:.6f}.png")
    cv2.imwrite(filename, cv_img)
    return filename

def interp_numeric(data_array, time_array, common_time):
    data_array = np.array(data_array)
    time_array = np.array(time_array)
    if len(data_array) == 0:
        return np.zeros((len(common_time),))
    if len(data_array.shape) == 1:
        return np.interp(common_time, time_array, data_array)
    else:
        interp_array = np.zeros((len(common_time), data_array.shape[1]))
        for i in range(data_array.shape[1]):
            interp_array[:, i] = np.interp(common_time, time_array, data_array[:, i])
        return interp_array

# ----------------------------
# Process each MCAP separately
# ----------------------------
for mcap_file in mcap_files:
    print(f"Processing {mcap_file} ...")

    # Local storage for this bag
    data = {
        "wrench": {topic: {"time": [], "force": [], "torque": []} for topic in topics["wrench"]},
        "ee_pose": {"time": [], "pos": [], "ori": []},
        "joint_states": {"time": [], "pos": [], "vel": [], "eff": []},
        "images": {topic: {"time": [], "filename": []} for topic in topics["images"]}
    }

    # Read bag
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            t = message.log_time / 1e9  # ns → s

            if channel.topic in topics["wrench"]:
                msg = deserialize_message(message.data, WrenchStamped)
                f_mag = np.linalg.norm([msg.wrench.force.x,
                                        msg.wrench.force.y,
                                        msg.wrench.force.z])
                tau_mag = np.linalg.norm([msg.wrench.torque.x,
                                          msg.wrench.torque.y,
                                          msg.wrench.torque.z])
                data["wrench"][channel.topic]["time"].append(t)
                data["wrench"][channel.topic]["force"].append(f_mag)
                data["wrench"][channel.topic]["torque"].append(tau_mag)

            elif channel.topic in topics["ee_pose"]:
                msg = deserialize_message(message.data, PoseStamped)
                data["ee_pose"]["time"].append(t)
                data["ee_pose"]["pos"].append([msg.pose.position.x,
                                               msg.pose.position.y,
                                               msg.pose.position.z])
                data["ee_pose"]["ori"].append([msg.pose.orientation.x,
                                               msg.pose.orientation.y,
                                               msg.pose.orientation.z,
                                               msg.pose.orientation.w])

            elif channel.topic in topics["joint_states"]:
                msg = deserialize_message(message.data, JointState)
                expected_joints = 9
                pos = list(msg.position)[:expected_joints]
                vel = list(msg.velocity)[:expected_joints]
                eff = list(msg.effort)[:expected_joints]
                while len(pos) < expected_joints: pos.append(0.0)
                while len(vel) < expected_joints: vel.append(0.0)
                while len(eff) < expected_joints: eff.append(0.0)
                data["joint_states"]["time"].append(t)
                data["joint_states"]["pos"].append(pos)
                data["joint_states"]["vel"].append(vel)
                data["joint_states"]["eff"].append(eff)

            elif channel.topic in topics["images"]:
                msg = deserialize_message(message.data, Image)
                prefix = channel.topic.split("/")[-2] + "_" + channel.topic.split("/")[-1]
                filename = save_image(msg, prefix, t)
                data["images"][channel.topic]["time"].append(t)
                data["images"][channel.topic]["filename"].append(filename)

    # ----------------------------
    # Interpolation setup
    # ----------------------------
    common_time = np.array(sorted(data["wrench"]["/franka_effort_real"]["time"]))
    if len(common_time) == 0:
        print(f"⚠️ No /franka_effort_real data in {mcap_file}, skipping")
        continue

    interp_wrench = {}
    for topic in topics["wrench"]:
        interp_wrench[topic] = {
            "force": interp_numeric(data["wrench"][topic]["force"], data["wrench"][topic]["time"], common_time),
            "torque": interp_numeric(data["wrench"][topic]["torque"], data["wrench"][topic]["time"], common_time)
        }

    ee_pos = interp_numeric(data["ee_pose"]["pos"], data["ee_pose"]["time"], common_time)
    ee_ori = interp_numeric(data["ee_pose"]["ori"], data["ee_pose"]["time"], common_time)
    joint_pos = interp_numeric(data["joint_states"]["pos"], data["joint_states"]["time"], common_time)
    joint_vel = interp_numeric(data["joint_states"]["vel"], data["joint_states"]["time"], common_time)
    joint_eff = interp_numeric(data["joint_states"]["eff"], data["joint_states"]["time"], common_time)

    image_links = {}
    for topic in topics["images"]:
        times = np.array(data["images"][topic]["time"])
        filenames = np.array(data["images"][topic]["filename"])
        if len(times) == 0:
            image_links[topic] = [""] * len(common_time)
        else:
            idx = np.searchsorted(times, common_time)
            idx[idx >= len(filenames)] = len(filenames) - 1
            image_links[topic] = filenames[idx]

    # ----------------------------
    # Save CSV (append mode)
    # ----------------------------
    file_tag = os.path.basename(mcap_file)

    # Compute force/torque differences (real - simulated)
    diff_force = interp_wrench["/franka_effort_real"]["force"] - interp_wrench["/franka_effort_sensor_to_hand"]["force"]
    diff_torque = interp_wrench["/franka_effort_real"]["torque"] - interp_wrench["/franka_effort_sensor_to_hand"]["torque"]

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)

        # Header (only once)
        if f.tell() == 0:
            header = ["file", "time"]
            for topic in topics["wrench"]:
                header += [f"{topic}_force", f"{topic}_torque"]
            # add new columns
            header += ["force_diff_real_vs_sim", "torque_diff_real_vs_sim"]
            header += ["ee_px", "ee_py", "ee_pz", "ee_ox", "ee_oy", "ee_oz", "ee_ow"]
            for j in range(joint_pos.shape[1]):
                header += [f"joint{j}_pos", f"joint{j}_vel", f"joint{j}_eff"]
            for topic in topics["images"]:
                header += [f"{topic}_img"]
            writer.writerow(header)

        # Rows
        for i in range(len(common_time)):
            row = [file_tag, common_time[i]]
            for topic in topics["wrench"]:
                row += [interp_wrench[topic]["force"][i], interp_wrench[topic]["torque"][i]]
            # differences
            row += [diff_force[i], diff_torque[i]]
            row += list(ee_pos[i]) + list(ee_ori[i])
            for j in range(joint_pos.shape[1]):
                row += [joint_pos[i, j], joint_vel[i, j], joint_eff[i, j]]
            for topic in topics["images"]:
                row += [image_links[topic][i]]
            writer.writerow(row)


print(f"✅ All files processed, merged CSV saved: {csv_filename}")
df = pd.read_csv(csv_filename)
print(df['file'].value_counts())
print(df.nunique())



""" #!/usr/bin/env python3

import os
import csv
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import cv2
from cv_bridge import CvBridge
import glob
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------

# parent_folders = ["/jazzy_ws/ros2bags_gearbox_1",
#         "/jazzy_ws/ros2bags_LObject_4"        
# ]

# mcap_files = []

# ignore_files = {"/jazzy_ws/ros2bags_LObject_4/test2/test2_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test3/test3_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test4/test4_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test5/test5_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test6/test6_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test7/test7_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test8/test8_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test9/test9_0.mcap",
#         "/jazzy_ws/ros2bags_LObject_4/test10/test10_0.mcap"
# }

# for parent_folder in parent_folders:
#     mcap_files_in_folder = glob.glob(os.path.join(parent_folder, "**", "*.mcap"), recursive = True)
#     mcap_files.extend(mcap_files_in_folder)

# mcap_files = [
#     f for f in mcap_files
#     if f not in ignore_files
# ]


# print(len(mcap_files))

mcap_files = [
    #"/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case5/test25/test25_0.mcap",
    "/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case2/test9/test9_0.mcap",
    #"/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case2/test10/test10_0.mcap",
    "/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case1/test1/test1_0.mcap"
]

# Topics
topics = {
    "wrench": [
        "/franka_effort_real",
        "/franka_effort_sensor_to_hand",
        "/franka_effort_link7_to_sensor"
    ],
    "ee_pose": ["/franka_EE_pose"],
    "joint_states": ["/joint_states"],
    "images": [
        "/floating_camera_rgb",
        "/floating_camera_depth",
        "/wrist_camera_rgb",
        "/wrist_camera_depth"
    ]
}

bridge = CvBridge()
image_folder = "rosbag_images_from_cameras"
os.makedirs(image_folder, exist_ok=True)

# ----------------------------
# Global storage (all bags together)
# ----------------------------
data = {
    "wrench": {topic: {"time": [], "force": [], "torque": []} for topic in topics["wrench"]},
    "ee_pose": {"time": [], "pos": [], "ori": []},
    "joint_states": {"time": [], "pos": [], "vel": [], "eff": []},
    "images": {topic: {"time": [], "filename": []} for topic in topics["images"]}
}

# ----------------------------
# Helper: save image and return filename
# ----------------------------
def save_image(msg, prefix, t):
    cv_img = bridge.imgmsg_to_cv2(msg)
    filename = os.path.join(image_folder, f"{prefix}_{t:.6f}.png")
    cv2.imwrite(filename, cv_img)
    return filename

# ----------------------------
# Process each MCAP (append to global data)
# ----------------------------
for mcap_file in mcap_files:
    print(f"Processing {mcap_file} ...")

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            t = message.log_time / 1e9  # ns -> s

            # Wrench
            if channel.topic in topics["wrench"]:
                msg = deserialize_message(message.data, WrenchStamped)
                f_mag = np.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2 + msg.wrench.force.z**2)
                tau_mag = np.sqrt(msg.wrench.torque.x**2 + msg.wrench.torque.y**2 + msg.wrench.torque.z**2)
                data["wrench"][channel.topic]["time"].append(t)
                data["wrench"][channel.topic]["force"].append(f_mag)
                data["wrench"][channel.topic]["torque"].append(tau_mag)

            # EE pose
            elif channel.topic in topics["ee_pose"]:
                msg = deserialize_message(message.data, PoseStamped)
                data["ee_pose"]["time"].append(t)
                data["ee_pose"]["pos"].append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
                data["ee_pose"]["ori"].append([msg.pose.orientation.x, msg.pose.orientation.y,
                                               msg.pose.orientation.z, msg.pose.orientation.w])

            # Joint states
            elif channel.topic in topics["joint_states"]:
                msg = deserialize_message(message.data, JointState)

                expected_joints = 9
                pos = list(msg.position)[:expected_joints]
                vel = list(msg.velocity)[:expected_joints]
                eff = list(msg.effort)[:expected_joints]

                while len(pos) < expected_joints: pos.append(0.0)
                while len(vel) < expected_joints: vel.append(0.0)
                while len(eff) < expected_joints: eff.append(0.0)

                data["joint_states"]["time"].append(t)
                data["joint_states"]["pos"].append(pos)
                data["joint_states"]["vel"].append(vel)
                data["joint_states"]["eff"].append(eff)

            # Images
            elif channel.topic in topics["images"]:
                msg = deserialize_message(message.data, Image)
                prefix = channel.topic.split("/")[-2] + "_" + channel.topic.split("/")[-1]
                filename = save_image(msg, prefix, t)
                data["images"][channel.topic]["time"].append(t)
                data["images"][channel.topic]["filename"].append(filename)

print(f"franka_effort_real force values: {data["wrench"]["/franka_effort_sensor_to_hand"]["force"]}")


# ----------------------------
# Interpolation setup
# ----------------------------
# Take all wrench times from /franka_effort_real as the common timeline
common_time = np.array(sorted(data["wrench"]["/franka_effort_real"]["time"]))
for t in common_time:
    print(f"common time: {t:.20f}")

def interp_numeric(data_array, time_array):
    data_array = np.array(data_array)
    time_array = np.array(time_array)
    if len(data_array) == 0:
        print("len is 0")
        return np.zeros((len(common_time),))
    if len(data_array.shape) == 1:
        print("len is 1")
        return np.interp(common_time, time_array, data_array)
    else:
        print("inside else")
        interp_array = np.zeros((len(common_time), data_array.shape[1]))
        for i in range(data_array.shape[1]):
            interp_array[:, i] = np.interp(common_time, time_array, np.array(data_array)[:, i])
        return interp_array

# Interpolate wrench
interp_wrench = {}
for topic in topics["wrench"]:
    interp_wrench[topic] = {
        "force": interp_numeric(data["wrench"][topic]["force"], data["wrench"][topic]["time"]),
        "torque": interp_numeric(data["wrench"][topic]["torque"], data["wrench"][topic]["time"])
    }

# Interpolate EE pose
ee_pos = interp_numeric(data["ee_pose"]["pos"], data["ee_pose"]["time"])
ee_ori = interp_numeric(data["ee_pose"]["ori"], data["ee_pose"]["time"])

# Interpolate joint states
joint_pos = interp_numeric(data["joint_states"]["pos"], data["joint_states"]["time"])
joint_vel = interp_numeric(data["joint_states"]["vel"], data["joint_states"]["time"])
joint_eff = interp_numeric(data["joint_states"]["eff"], data["joint_states"]["time"])

# Match images to closest time
image_links = {}
for topic in topics["images"]:
    times = np.array(data["images"][topic]["time"])
    filenames = np.array(data["images"][topic]["filename"])
    if len(times) == 0:
        image_links[topic] = [""] * len(common_time)
    else:
        idx = np.searchsorted(times, common_time)
        idx[idx >= len(filenames)] = len(filenames) - 1
        image_links[topic] = filenames[idx]


print(f"Interpolated wrench force: {interp_wrench["/franka_effort_sensor_to_hand"]["force"]}")
# ----------------------------
# Save one merged CSV
# ----------------------------
csv_filename = "data_from_all_rosbags.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    # Header
    header = ["time"]
    for topic in topics["wrench"]:
        header += [f"{topic}_force", f"{topic}_torque"]
    header += ["ee_px", "ee_py", "ee_pz", "ee_ox", "ee_oy", "ee_oz", "ee_ow"]
    for j in range(joint_pos.shape[1]):
        header += [f"joint{j}_pos", f"joint{j}_vel", f"joint{j}_eff"]
    for topic in topics["images"]:
        header += [f"{topic}_img"]
    writer.writerow(header)

    # Rows
    for i in range(len(common_time)):
        row = [common_time[i]]
        for topic in topics["wrench"]:
            row += [interp_wrench[topic]["force"][i], interp_wrench[topic]["torque"][i]]
        row += list(ee_pos[i]) + list(ee_ori[i])
        for j in range(joint_pos.shape[1]):
            row += [joint_pos[i, j], joint_vel[i, j], joint_eff[i, j]]
        for topic in topics["images"]:
            row += [image_links[topic][i]]
        writer.writerow(row)

print(f"Merged CSV saved: {csv_filename}")
df = pd.read_csv("data_from_all_rosbags.csv")
print(df.nunique())

 """
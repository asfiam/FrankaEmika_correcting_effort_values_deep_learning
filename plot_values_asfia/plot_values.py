#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import csv
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import WrenchStamped


#mcap_file = "/home/digitaltwin/IsaacSim-ros_workspaces/jazzy_ws/ros2bags_LObject/test1/test1_0.mcap"
#mcap_file = "/jazzy_ws/ros2bags_LObject_4/LObject_4_Case6/test35/test35_0.mcap"
#mcap_files = ["/jazzy_ws/test26/test26_0.mcap"]
mcap_files = ["/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case5/test25/test25_0.mcap",
            #"/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case2/test9/test9_0.mcap",
            #"/jazzy_ws/ros2bags_gearbox_1/gearbox_1_Case2/test10/test10_0.mcap"
]

topics = [
    "/franka_effort_real",
    "/franka_effort_sensor_to_hand",
    "/franka_effort_link7_to_sensor"
]
  
data = {
    topic: {"time": [], "force": [], "torque": []}
    for topic in topics
}

# data = {topic: {"time": []} for topic in topics}

# For WrenchStamped topics, also store force/torque
for topic in ["/franka_effort_real", "/franka_effort_sensor_to_hand", "/franka_effort_link7_to_sensor"]:
    data[topic].update({"force": [], "torque": []})

for mcap_file in mcap_files:
    print(f"reading file {mcap_file}")
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=topics):
            t = message.log_time / 1e9  # ns to s
            msg = deserialize_message(message.data, WrenchStamped)

            f_mag = np.sqrt(
                msg.wrench.force.x**2 +
                msg.wrench.force.y**2 +
                msg.wrench.force.z**2
            )

            tau_mag = np.sqrt(
                msg.wrench.torque.x**2 +
                msg.wrench.torque.y**2 +
                msg.wrench.torque.z**2
            )

            data[channel.topic]["time"].append(t)
            data[channel.topic]["force"].append(f_mag)
            data[channel.topic]["torque"].append(tau_mag)

# for mcap_file in mcap_files:
#     print(f"reading file {mcap_file}")
#     with open(mcap_file, "rb") as f:
#         reader = make_reader(f)
#         for schema, channel, message in reader.iter_messages(topics=topics):
#             t = message.log_time / 1e9

#             if channel.topic in ["/franka_effort_real", "/franka_effort_sensor_to_hand", "/franka_effort_link7_to_sensor"]:
#                 msg = deserialize_message(message.data, WrenchStamped)
#                 f_mag = np.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2 + msg.wrench.force.z**2)
#                 tau_mag = np.sqrt(msg.wrench.torque.x**2 + msg.wrench.torque.y**2 + msg.wrench.torque.z**2)
#                 data[channel.topic]["time"].append(t)
#                 data[channel.topic]["force"].append(f_mag)
#                 data[channel.topic]["torque"].append(tau_mag)

#             elif channel.topic == "/franka_ee_pose":
#                 msg = deserialize_message(message.data, PoseStamped)
#                 pos = msg.pose.position
#                 ori = msg.pose.orientation
#                 data[channel.topic]["time"].append(t)
#                 data[channel.topic]["pos"] = data[channel.topic].get("pos", []) + [[pos.x, pos.y, pos.z]]
#                 data[channel.topic]["ori"] = data[channel.topic].get("ori", []) + [[ori.x, ori.y, ori.z, ori.w]]

#             elif channel.topic == "/franka_joint_states":
#                 msg = deserialize_message(message.data, JointState)
#                 data[channel.topic]["time"].append(t)
#                 data[channel.topic]["pos"] = data[channel.topic].get("pos", []) + [msg.position]
#                 data[channel.topic]["vel"] = data[channel.topic].get("vel", []) + [msg.velocity]
#                 data[channel.topic]["effort"] = data[channel.topic].get("effort", []) + [msg.effort]

#             elif channel.topic in ["/camera_floating/rgb", "/camera_floating/depth",
#                                 "/camera_wrist/rgb", "/camera_wrist/depth"]:
#                 msg = deserialize_message(message.data, Image)
#                 img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1)) if msg.encoding in ["rgb8", "rgba8"] else np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
                
#                 folder = f"images/{channel.topic.replace('/', '_')}"
#                 os.makedirs(folder, exist_ok=True)
#                 filename = os.path.join(folder, f"{t:.6f}.png")
#                 cv2.imwrite(filename, img_array)
#                 data[channel.topic]["time"].append(t)
#                 data[channel.topic]["file"] = data[channel.topic].get("file", []) + [filename]




for topic in topics:
    data[topic]["time"] = np.array(data[topic]["time"])
    data[topic]["force"] = np.array(data[topic]["force"])
    data[topic]["torque"] = np.array(data[topic]["torque"])


common_time = data["/franka_effort_real"]["time"]

#interpolation
sensor_to_hand_force = np.interp(common_time,
                             data["/franka_effort_sensor_to_hand"]["time"],
                             data["/franka_effort_sensor_to_hand"]["force"])

sensor_to_hand_torque = np.interp(common_time,
                                   data["/franka_effort_sensor_to_hand"]["time"],
                                   data["/franka_effort_sensor_to_hand"]["torque"])

link7_to_sensor_force = np.interp(common_time,
                                    data["/franka_effort_link7_to_sensor"]["time"],
                                    data["/franka_effort_link7_to_sensor"]["force"])
                                    
link7_to_sensor_torque = np.interp(common_time,
                                    data["/franka_effort_link7_to_sensor"]["time"],
                                    data["/franka_effort_link7_to_sensor"]["torque"])
                                    
#max-min normalization        
# sensor_to_hand_force = (sensor_to_hand_force - sensor_to_hand_force.min()) / (sensor_to_hand_force.max() - sensor_to_hand_force.min()) * ((data["/franka_effort_real"]["force"]).max() - (data["/franka_effort_real"]["force"]).min()) + (data["/franka_effort_real"]["force"]).min()
# link7_to_sensor_force = (link7_to_sensor_force - link7_to_sensor_force.min()) / (link7_to_sensor_force.max() - link7_to_sensor_force.min()) * ((data["/franka_effort_real"]["force"]).max() - (data["/franka_effort_real"]["force"]).min()) + (data["/franka_effort_real"]["force"]).min()


csv_filename = "effort_values.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "time",
        "real_force", "sensor to hand force", "link7 to sensor force",
        "real_torque", "sensor to hand torque", "link7 to sensor torque"
    ])
    for i in range(len(common_time)):
        writer.writerow([
            common_time[i],
            data["/franka_effort_real"]["force"][i],
            sensor_to_hand_force[i],
            link7_to_sensor_force[i],
            data["/franka_effort_real"]["torque"][i],
            sensor_to_hand_torque[i],
            link7_to_sensor_torque[i]
            
        ])

print(f"CSV file saved as {csv_filename}")


plt.figure(figsize=(12, 5))
plt.plot(common_time, data["/franka_effort_real"]["force"], label="Real Force")
plt.plot(common_time, (sensor_to_hand_force) , label="Sim Force (sensor to hand)")
plt.plot(common_time, (link7_to_sensor_force), label="Sim Force (link7 to sensor)")
plt.xlabel("Time [s]")
plt.ylabel("Force Magnitude [N]")
plt.title("Force Magnitude Comparison")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(common_time, data["/franka_effort_real"]["torque"], label="Real Torque")
plt.plot(common_time, sensor_to_hand_torque, label="Sim Torque (sensor to hand)")
plt.plot(common_time, link7_to_sensor_torque, label="Sim Torque (link7 to sensor)")
plt.xlabel("Time [s]")
plt.ylabel("Torque Magnitude [Nm]")
plt.title("Torque Magnitude Comparison")
plt.legend()
plt.grid(True)
plt.show()


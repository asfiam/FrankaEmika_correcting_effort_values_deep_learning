#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "neural_net_franka_sequential"))

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import cv2
import pandas as pd
from collections import deque

from std_msgs.msg import Float32MultiArray

from neural_net_franka_sequential.model_seq import CNNLSTMNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 4
NUMERIC_DIM = 36   # (2 sim_effort + 7 pose + 27 joint states)
OUTPUT_DIM = 2     # (force_mag, torque_mag)

class LiveInferenceNode(Node):
    def __init__(self, csv_path=None, img_root=None, test_mode=False):
        super().__init__("live_inference_node")

        # --- Model ---
        self.model = CNNLSTMNet(
            numeric_dim=NUMERIC_DIM,
            output_dim=OUTPUT_DIM
        ).to(DEVICE)
        self.model.load_state_dict(torch.load("cnn_lstm_franka_residual.pth", map_location=DEVICE))
        self.model.eval()
        self.get_logger().info("✅ Model loaded and ready for live inference")

        # --- Buffers ---
        self.numeric_buffer = deque(maxlen=SEQ_LEN)
        self.image_buffer = deque(maxlen=SEQ_LEN)

        # --- Publisher ---
        self.pub_pred = self.create_publisher(Float32MultiArray, "/predicted_real_ft_magnitudes", 10)

        # --- CSV test mode ---
        self.test_mode = test_mode
        if test_mode:
            assert csv_path is not None and img_root is not None, "Need CSV path and image folder for test mode"
            self.df = pd.read_csv(csv_path)
            self.img_root = img_root
            self.row_idx = 0
            # timer to simulate incoming data
            self.create_timer(0.1, self.feed_from_csv)

    # ---------- Dummy feed from CSV ----------
    def feed_from_csv(self):
        if self.row_idx >= len(self.df):
            self.get_logger().info("✅ Finished reading all rows from CSV")
            return

        row = self.df.iloc[self.row_idx]

        # simulated effort (force + torque from sensor_to_hand)
        sim_force = float(row["/franka_effort_sensor_to_hand_force"])
        sim_torque = float(row["/franka_effort_sensor_to_hand_torque"])
        latest_sim_effort = np.array([sim_force, sim_torque], dtype=np.float32)

        # pose (7 values)
        pose = np.array([
            row["ee_px"], row["ee_py"], row["ee_pz"],
            row["ee_ox"], row["ee_oy"], row["ee_oz"], row["ee_ow"]
        ], dtype=np.float32)

        # joints (9×3 = 27 values)
        joints = []
        for j in range(9):
            joints.extend([
                row[f"joint{j}_pos"],
                row[f"joint{j}_vel"],
                row[f"joint{j}_eff"]
            ])
        joints = np.array(joints, dtype=np.float32)

        numeric_vec = np.concatenate([latest_sim_effort, pose, joints])
        numeric_tensor = torch.tensor(numeric_vec, dtype=torch.float32)

        # load images
        def load_img(path, rgb=True):
            img = cv2.imread(os.path.join(self.img_root, os.path.basename(path)), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (128, 128))
            if rgb:
                if img.ndim == 2:  # grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                if img.ndim == 2:
                    img = img[..., None]
            return torch.tensor(img, dtype=torch.float32).permute(2,0,1) / 255.0

        stacked_img = torch.cat([
            load_img(row["/floating_camera_rgb_img"], rgb=True),
            load_img(row["/floating_camera_depth_img"], rgb=False),
            load_img(row["/wrist_camera_rgb_img"], rgb=True),
            load_img(row["/wrist_camera_depth_img"], rgb=False)
        ], dim=0)

        # update buffers
        self.numeric_buffer.append(numeric_tensor)
        self.image_buffer.append(stacked_img)

        if len(self.numeric_buffer) == SEQ_LEN:
            self.run_inference(latest_sim_effort)

        self.row_idx += 1

    # ---------- Inference ----------
    def run_inference(self, sim_effort):
        numeric_seq = torch.stack(list(self.numeric_buffer)).unsqueeze(0).to(DEVICE)  # (1, S, 36)
        image_seq = torch.stack(list(self.image_buffer)).unsqueeze(0).to(DEVICE)      # (1, S, 8, 128, 128)

        with torch.no_grad():
            residual_pred = self.model(numeric_seq, image_seq)  # (1, 2)

        corrected = sim_effort + residual_pred.cpu().numpy().flatten()

        # publish Float32MultiArray
        msg = Float32MultiArray()
        msg.data = corrected.tolist()
        self.pub_pred.publish(msg)

        self.get_logger().info(f"[Row {self.row_idx}] Predicted Real Force={corrected[0]:.3f}, Torque={corrected[1]:.3f}")


def main(args=None):
    rclpy.init(args=args)

    # ⚡ Switch between modes here:
    test_mode = True   # <-- set False when subscribing to ROS topics
    if test_mode:
        csv_path = "data_from_all_rosbags.csv"
        img_root = "rosbag_images_from_cameras"
        node = LiveInferenceNode(csv_path=csv_path, img_root=img_root, test_mode=True)
    else:
        node = LiveInferenceNode(test_mode=False)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

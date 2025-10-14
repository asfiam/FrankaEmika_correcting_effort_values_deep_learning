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
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from cv_bridge import CvBridge

from neural_net_franka_sequential.model_seq import CNNLSTMNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 4
NUMERIC_DIM = 36
OUTPUT_DIM = 2


class LiveInferenceNode(Node):
    def __init__(self, csv_path=None, img_root=None, test_mode=False):
        super().__init__("live_inference_node")
        self.test_mode = test_mode

        # --- Model ---
        self.model = CNNLSTMNet(numeric_dim=NUMERIC_DIM, output_dim=OUTPUT_DIM).to(DEVICE)
        self.model.load_state_dict(torch.load("cnn_lstm_franka_residual.pth", map_location=DEVICE))
        self.model.eval()
        self.get_logger().info("✅ Model loaded and ready for inference")

        # --- Check for NaNs in model weights ---
        nan_in_weights = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                self.get_logger().error(f"❌ NaN found in layer {name}")
                nan_in_weights = True
        if not nan_in_weights:
            self.get_logger().info("✅ No NaNs found in model weights")

        # --- Forward hook to catch NaNs/Inf during runtime ---
        # def hook_fn(module, input, output):
        #     if isinstance(output, tuple):
        #         for i, o in enumerate(output):
        #             if torch.isnan(o).any() or torch.isinf(o).any():
        #                 self.get_logger().warn(f"⚠️ NaN/Inf in {module.__class__.__name__} output[{i}] "
        #                                         f"(mean={o.mean().item():.4f}, std={o.std().item():.4f})")
        #     elif torch.isnan(output).any() or torch.isinf(output).any():
        #         self.get_logger().warn(f"⚠️ NaN/Inf in {module.__class__.__name__} "
        #                                 f"(mean={output.mean().item():.4f}, std={output.std().item():.4f})")

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
        #         module.register_forward_hook(hook_fn)

        # --- Dummy forward test ---
        try:
            self.get_logger().info("🧪 Running dummy forward-pass check...")
            dummy_num = torch.zeros((1, SEQ_LEN, NUMERIC_DIM), device=DEVICE)
            dummy_img = torch.zeros((1, SEQ_LEN, 8, 128, 128), device=DEVICE)
            with torch.no_grad():
                dummy_out = self.model(dummy_num, dummy_img)

            # Handle both single tensor and tuple outputs
            if isinstance(dummy_out, tuple):
                dummy_out = dummy_out[0]
                all_finite = True
                for i, out in enumerate(dummy_out):
                    if not isinstance(out, torch.Tensor):
                        self.get_logger().warn(f"⚠️ Output[{i}] is not a tensor (type={type(out)}). Skipping check.")
                        continue
                    if not torch.isfinite(out).all():
                        self.get_logger().error(f"❌ Dummy forward produced NaN/Inf in output[{i}]")
                        all_finite = False
                    else:
                        self.get_logger().info(f"✅ Output[{i}] OK (mean={out.mean().item():.4f}, std={out.std().item():.4f})")
                if all_finite:
                    self.get_logger().info("✅ Dummy forward successful (tuple outputs all finite)")
                else:
                    self.get_logger().error("❌ Dummy forward produced invalid outputs in tuple")
            else:
                if torch.isfinite(dummy_out).all():
                    self.get_logger().info(f"✅ Dummy forward successful (single output OK). "
                                        f"Mean={dummy_out.mean().item():.4f}, Std={dummy_out.std().item():.4f}")
                else:
                    self.get_logger().error("❌ Dummy forward produced NaN/Inf values.")
        except Exception as e:
            self.get_logger().error(f"❌ Dummy forward failed: {e}")

        # --- Buffers for sequential input ---
        self.numeric_buffer = deque(maxlen=SEQ_LEN)
        self.image_buffer = deque(maxlen=SEQ_LEN)

        # --- Publisher for predicted magnitudes ---
        self.pub_pred = self.create_publisher(Float32MultiArray, "/predicted_real_ft_magnitudes", 10)
        self.pub_sim = self.create_publisher(Float32MultiArray, "/simulated_ft_magnitudes", 10)


        # --- Mode selection ---
        #self.test_mode = test_mode.lower()
        self.bridge = CvBridge()

        if self.test_mode:
            assert csv_path and img_root, "Need CSV path and image folder for test mode"
            self.df = pd.read_csv(csv_path)
            self.img_root = img_root
            self.row_idx = 0
            self.create_timer(0.1, self.feed_from_csv)
            self.get_logger().info("🧪 Running in CSV test mode.")

        elif self.test_mode == False:
            # Same pipeline — ROS bag and live mode both rely on topic data
            self.sim_effort = None
            self.pose = None
            self.joints = None
            self.images = {}

            self.create_subscription(WrenchStamped, "/franka_effort_sensor_to_hand", self.sim_effort_cb, 10)
            self.create_subscription(PoseStamped, "/franka_EE_pose", self.pose_cb, 10)
            self.create_subscription(JointState, "/joint_states", self.joints_cb, 10)

            self.create_subscription(Image, "/floating_camera_rgb", lambda msg: self.image_cb(msg, "floating_rgb"), 10)
            self.create_subscription(Image, "/floating_camera_depth", lambda msg: self.image_cb(msg, "floating_depth"), 10)
            self.create_subscription(Image, "/wrist_camera_rgb", lambda msg: self.image_cb(msg, "wrist_rgb"), 10)
            self.create_subscription(Image, "/wrist_camera_depth", lambda msg: self.image_cb(msg, "wrist_depth"), 10)

            self.create_timer(0.1, self.try_inference_from_topics)
            self.get_logger().info(f"📡 Running in {'rosbag' if self.test_mode == 'rosbag' else 'live'} mode.")
        else:
            self.get_logger().warn(f"⚠️ Unknown test mode '{test_mode}', defaulting to live mode.")


    # ---------- CSV MODE ----------
    def feed_from_csv(self):
        if self.row_idx >= len(self.df):
            self.get_logger().info("✅ Finished reading all rows from CSV")
            return

        row = self.df.iloc[self.row_idx]

        sim_force = float(row["/franka_effort_sensor_to_hand_force"])
        sim_torque = float(row["/franka_effort_sensor_to_hand_torque"])
        latest_sim_effort = np.array([sim_force, sim_torque], dtype=np.float32)

        pose = np.array([row["ee_px"], row["ee_py"], row["ee_pz"],
                         row["ee_ox"], row["ee_oy"], row["ee_oz"], row["ee_ow"]], dtype=np.float32)

        joints = np.array([row[f"joint{j}_{k}"] for j in range(9) for k in ["pos", "vel", "eff"]], dtype=np.float32)

        numeric_vec = np.concatenate([latest_sim_effort, pose, joints])
        numeric_tensor = torch.tensor(numeric_vec, dtype=torch.float32)

        def load_img(path, rgb=True):
            img = cv2.imread(os.path.join(self.img_root, os.path.basename(path)), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (128, 128))
            if rgb and img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif not rgb and img.ndim == 2:
                img = img[..., None]
            return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        stacked_img = torch.cat([
            load_img(row["/floating_camera_rgb_img"], rgb=True),
            load_img(row["/floating_camera_depth_img"], rgb=False),
            load_img(row["/wrist_camera_rgb_img"], rgb=True),
            load_img(row["/wrist_camera_depth_img"], rgb=False)
        ], dim=0)

        self.numeric_buffer.append(numeric_tensor)
        self.image_buffer.append(stacked_img)

        if len(self.numeric_buffer) == SEQ_LEN:
            self.run_inference(numeric_vec, latest_sim_effort)

        self.row_idx += 1

    # ---------- ROS MODE ----------
    def sim_effort_cb(self, msg: WrenchStamped):
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        tx, ty, tz = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        force_mag = np.sqrt(fx**2 + fy**2 + fz**2)
        torque_mag = np.sqrt(tx**2 + ty**2 + tz**2)
        self.sim_effort = np.array([force_mag, torque_mag], dtype=np.float32)

    def pose_cb(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        self.pose = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)

    def joints_cb(self, msg: JointState):
        self.get_logger().info(f"Received joints: pos={len(msg.position)}, vel={len(msg.velocity)}, eff={len(msg.effort)}")
        self.joints = np.array([msg.position[i] for i in range(9)] +
                               [msg.velocity[i] for i in range(9)] +
                               [msg.effort[i] for i in range(9)], dtype=np.float32)

    # def image_cb(self, msg: Image, key: str):
    #     cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #     cv_img = cv2.resize(cv_img, (128, 128))
    #     self.get_logger().info(f"📸 {key} image dtype={cv_img.dtype}, range=({np.nanmin(cv_img)}, {np.nanmax(cv_img)})")

    #     if "rgb" in key and cv_img.ndim == 2:
    #         cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    #     elif "depth" in key and cv_img.ndim == 2:
    #         cv_img = cv_img[..., None]
    #     self.images[key] = torch.tensor(cv_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

    # def image_cb(self, msg: Image, key: str):
    #     # Convert ROS Image message to OpenCV format
    #     cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #     cv_img = cv2.resize(cv_img, (128, 128))

    #     # Log initial stats
    #     self.get_logger().info(
    #         f"📸 {key}: dtype={cv_img.dtype}, shape={cv_img.shape}, range=({np.nanmin(cv_img)}, {np.nanmax(cv_img)})"
    #     )

    #     # Handle grayscale or RGBA images
    #     if cv_img.ndim == 2:
    #         cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    #         color_status = "gray→BGR"
    #     elif cv_img.shape[2] == 4:
    #         cv_img = cv_img[:, :, :3]
    #         color_status = "RGBA→RGB"
    #     else:
    #         color_status = "3-channel input"

    #     # Detect likely color order before conversion
    #     avg_r = np.mean(cv_img[:, :, 0])
    #     avg_g = np.mean(cv_img[:, :, 1])
    #     avg_b = np.mean(cv_img[:, :, 2])

    #     # Heuristic: if red channel average is highest, it’s probably RGB
    #     if avg_r > avg_b:
    #         original_order = "RGB"
    #     else:
    #         original_order = "BGR"

    #     # Force convert to BGR
    #     cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
    #     self.get_logger().info(
    #         f"🎨 {key}: detected {original_order}, forced to BGR ({color_status})"
    #     )

    #     # Convert to torch tensor
    #     self.images[key] = (
    #         torch.tensor(cv_img_bgr, dtype=torch.float32).permute(2, 0, 1) / 255.0
    #     )

    # def image_cb(self, msg: Image, key: str):
    #     cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #     cv_img = cv2.resize(cv_img, (128, 128))
    #     print("I am here 1")
    #     # Log basic info
    #     self.get_logger().info(
    #         f"📸 {key}: dtype={cv_img.dtype}, shape={cv_img.shape}, "
    #         f"range=({np.nanmin(cv_img)}, {np.nanmax(cv_img)})"
    #     )

    #     # Replace invalid depth values with 0 (or another reasonable value)
    #     if "depth" in key:
    #         print("I am here 2")
    #         cv_img = np.nan_to_num(cv_img, nan=0.0, posinf=0.0, neginf=0.0)
    #         # Expand to 3 channels for consistency (so later stacking works)
    #         cv_img = cv2.cvtColor(cv_img.astype(np.float32), cv2.COLOR_GRAY2BGR)

    #     elif "rgb" in key:
    #         print("I am here 3")
    #         if cv_img.ndim == 2:
    #             cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    #         else:
    #             # Force to BGR
    #             cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    #     # Final conversion to tensor
    #     print("I am here 4")
    #     tensor_img = torch.tensor(cv_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

    #     # Safety check
    #     if not torch.isfinite(tensor_img).all():
    #         invalid = torch.nonzero(~torch.isfinite(tensor_img))
    #         self.get_logger().error(
    #             f"❌ NaN/Inf found in {key}! Indices: {invalid[:10]} "
    #             f"(showing first 10 of {invalid.size(0)})"
    #         )

    #     self.images[key] = tensor_img


    def image_cb(self, msg: Image, key: str):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv_img = cv2.resize(cv_img, (128, 128))
        print("I am here 1")

        # Log basic info
        self.get_logger().info(
            f"📸 {key}: dtype={cv_img.dtype}, shape={cv_img.shape}, "
            f"range=({np.nanmin(cv_img)}, {np.nanmax(cv_img)})"
        )

        if "depth" in key:
            print("I am here 2")
            # Replace invalid depth values with 0
            cv_img = np.nan_to_num(cv_img, nan=0.0, posinf=0.0, neginf=0.0)
            # Ensure single-channel (H, W, 1) for stacking
            if cv_img.ndim == 2:
                cv_img = cv_img[..., None]

        # elif "rgb" in key:
        #     print("I am here 3")
        #     if cv_img.ndim == 2:
        #         cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        #     else:
        #         # Force to BGR
        #         cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        # Final conversion to tensor
        print("I am here 4")
        tensor_img = torch.tensor(cv_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Safety check
        if not torch.isfinite(tensor_img).all():
            invalid = torch.nonzero(~torch.isfinite(tensor_img))
            self.get_logger().error(
                f"❌ NaN/Inf found in {key}! Indices: {invalid[:10]} "
                f"(showing first 10 of {invalid.size(0)})"
            )

        self.images[key] = tensor_img


    def try_inference_from_topics(self):
        #print("inside try inference from topics")
        if self.sim_effort is None or self.pose is None or self.joints is None:
            return
        if not all(k in self.images for k in ["floating_rgb", "floating_depth", "wrist_rgb", "wrist_depth"]):
            return

        numeric_vec = np.concatenate([self.sim_effort, self.pose, self.joints])
        numeric_tensor = torch.tensor(numeric_vec, dtype=torch.float32)

        stacked_img = torch.cat([
            self.images["floating_rgb"],
            self.images["floating_depth"],
            self.images["wrist_rgb"],
            self.images["wrist_depth"]
        ], dim=0)


        #stacked_img = torch.zeros((8, 128, 128), dtype=torch.float32)

        #print("i am here")
        self.numeric_buffer.append(numeric_tensor)
        self.image_buffer.append(stacked_img)

        if len(self.numeric_buffer) >= SEQ_LEN:
            self.run_inference(numeric_vec, self.sim_effort)

    # ---------- Inference ----------
    def run_inference(self, numeric_vec, sim_effort):
        if len(self.numeric_buffer) < SEQ_LEN or len(self.image_buffer) < SEQ_LEN:
            self.get_logger().warn(f"Skipping inference — incomplete buffer (numeric={len(self.numeric_buffer)}, image={len(self.image_buffer)})")

        numeric_seq = torch.stack(list(self.numeric_buffer)).float().unsqueeze(0).to(DEVICE)
        image_seq = torch.stack(list(self.image_buffer)).float().unsqueeze(0).to(DEVICE)
        #print (f"image seq: {image_seq}")
        # 🔍 Input validation
        if not torch.isfinite(numeric_seq).all():
            bad_indices = torch.isnan(numeric_seq) | torch.isinf(numeric_seq)
            self.get_logger().error(f"❌ NaN/Inf found in numeric buffer! Indices: {torch.nonzero(bad_indices)}")
            self.get_logger().error(f"Values: {numeric_seq[bad_indices]}")
            return

        if not torch.isfinite(image_seq).all():
            bad_indices = torch.isnan(image_seq) | torch.isinf(image_seq)
            self.get_logger().error(f"❌ NaN/Inf found in image buffer! Indices: {torch.nonzero(bad_indices)}")
            self.get_logger().error(f"Values: {image_seq[bad_indices]}")
            return

        self.get_logger().info(f"Sequence buffer size: {len(self.image_buffer)} (expected {SEQ_LEN})")

        # 🔮 Forward pass with range and NaN checks
        with torch.no_grad():
            output = self.model(numeric_seq, image_seq)

        self.get_logger().info(f"🔎 Output range before clamp: [{torch.min(output).item()}, {torch.max(output).item()}]")
        if torch.isnan(output).any() or torch.isinf(output).any():
            self.get_logger().error("❌ NaNs/Inf detected in model output!")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)

        output = torch.clamp(output, min=-1000, max=1000)

        mean_val, std_val, max_val = output.mean().item(), output.std().item(), torch.max(torch.abs(output)).item()
        self.get_logger().info(f"Output stats — mean: {mean_val:.4f}, std: {std_val:.4f}, max: {max_val:.2f}")

        corrected = sim_effort + output.cpu().numpy().flatten()
        msg = Float32MultiArray()
        msg.data = corrected.tolist()
        self.pub_pred.publish(msg)

        msg_sim = Float32MultiArray()
        msg_sim.data = sim_effort.tolist()
        self.pub_sim.publish(msg_sim)

        self.get_logger().info(f"Pred Real Force={corrected[0]:.3f}, Torque={corrected[1]:.3f} | Sim Force={sim_effort[0]:.3f}, Torque={sim_effort[1]:.3f}")

        #self.get_logger().info(f"Pred Real Force={corrected[0]:.3f}, Torque={corrected[1]:.3f}")



def main(args=None):
    rclpy.init(args=args)
    test_mode = False
    if test_mode:
        node = LiveInferenceNode(csv_path="data_from_all_rosbags.csv", img_root="rosbag_images_from_cameras", test_mode=True)
    else:
        node = LiveInferenceNode(test_mode=False)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

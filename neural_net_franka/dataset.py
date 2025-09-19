import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

class RosbagDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.numeric_cols = [
            "/franka_effort_sensor_to_hand_force",
            "/franka_effort_sensor_to_hand_torque",
            "ee_px", "ee_py", "ee_pz",
            "ee_ox", "ee_oy", "ee_oz", "ee_ow",
            "joint0_pos", "joint0_vel", "joint0_eff",
            "joint1_pos", "joint1_vel", "joint1_eff",
            "joint2_pos", "joint2_vel", "joint2_eff",
            "joint3_pos", "joint3_vel", "joint3_eff",
            "joint4_pos", "joint4_vel", "joint4_eff",
            "joint5_pos", "joint5_vel", "joint5_eff",
            "joint6_pos", "joint6_vel", "joint6_eff",
            "joint7_pos", "joint7_vel", "joint7_eff",
            "joint8_pos", "joint8_vel", "joint8_eff",
        ]

        self.image_cols = [
            "/floating_camera_rgb_img",
            "/floating_camera_depth_img",          
            "/wrist_camera_rgb_img",
            "/wrist_camera_depth_img"
        ]
        
        self.target_cols = [
            "/franka_effort_real_force",
            "/franka_effort_real_torque"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # numeric features
        numeric = torch.tensor(row[self.numeric_cols].astype(float).values, dtype=torch.float32)

        # images
        images = []
        for col in self.image_cols:
            img_path = row[col]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (128, 128))
            if img.ndim == 2:
                img = img[..., None]
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            images.append(img)

        images = torch.cat(images, dim=0)  

        # real effort (ground truth)
        target = torch.tensor(row.loc[self.target_cols].astype(float).values, dtype=torch.float32)

        return numeric, images, target

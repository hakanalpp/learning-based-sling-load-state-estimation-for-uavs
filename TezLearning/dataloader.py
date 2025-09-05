import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import utils


class StateBasedDroneDataset(Dataset):
    def __init__(self, images_folder, csv_path=None, csv_data=None):
        if csv_data is not None:
            self.df = csv_data.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path, dtype={"frameId": str})
        else:
            raise ValueError("Must provide either csv_path or csv_data")
        
        self.df["frameId"] = self.df["frameId"].astype(str)
        self.images_folder = images_folder

        self.samples = []
        for _, row in self.df.iterrows():
            frame_id = row["frameId"]
            img_path = os.path.join(images_folder, f"{frame_id}.jpg")

            if not os.path.exists(img_path):
                continue

            drone_pos = np.array(
                [row["drone_pos_x"], row["drone_pos_y"], row["drone_pos_z"]]
            )
            camera_rot = np.array(
                [
                    row["camera_rot_x"],
                    row["camera_rot_y"],
                    row["camera_rot_z"],
                    row["camera_rot_w"],
                ]
            )
            cargo_pos = np.array(
                [row["cargo_pos_x"], row["cargo_pos_y"], row["cargo_pos_z"]]
            )
            cargo_rot = np.array(
                [
                    row["cargo_rot_x"],
                    row["cargo_rot_y"],
                    row["cargo_rot_z"],
                    row["cargo_rot_w"],
                ]
            )
            cargo_vel = np.array(
                [row["cargo_vel_x"], row["cargo_vel_y"], row["cargo_vel_z"]]
            )

            cam_R = Rotation.from_quat(camera_rot)
            vec_world = cargo_pos - drone_pos
            vec_cam = cam_R.inv().apply(vec_world)
            distance = np.linalg.norm(vec_world)

            if distance < 1e-6:
                continue

            # Relative rotation and velocity in camera frame
            cargo_R_world = Rotation.from_quat(cargo_rot)
            cargo_R_cam = cam_R.inv() * cargo_R_world
            cargo_vel_cam = cam_R.inv().apply(cargo_vel)

            # Create label: [direction(3), distance(1), rotation(4), velocity(3)]
            label = np.concatenate(
                [
                    vec_cam / distance,  # normalized direction
                    [distance],
                    cargo_R_cam.as_quat(),
                    cargo_vel_cam,
                ]
            )

            self.samples.append(
                {
                    "frame_id": frame_id,
                    "image_path": img_path,
                    "label": torch.tensor(label, dtype=torch.float32),
                }
            )

        print(f"Loaded {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(sample["image_path"])
        img_tensor = utils.preprocess_image(img)

        return {
            "frame_id": sample["frame_id"],
            "image": img_tensor,
            "label": sample["label"],
        }
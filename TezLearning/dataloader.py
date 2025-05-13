import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import utils


class CargoDataset(Dataset):
    def __init__(self, images_folder, csv_path):
        self.df = pd.read_csv(csv_path, dtype={"frameId": str})
        self.images_folder = images_folder

        self.samples = []
        for _, row in self.df.iterrows():
            frame_id = row["frameId"]
            img_file = os.path.join(images_folder, f"{frame_id}.jpg")

            drone_pos = np.array(
                [row["drone_pos_x"], row["drone_pos_y"], row["drone_pos_z"]]
            )
            drone_rot = np.array(
                [
                    row["drone_rot_x"],
                    row["drone_rot_y"],
                    row["drone_rot_z"],
                    row["drone_rot_w"],
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

            vec_world = cargo_pos - drone_pos
            vec_local = Rotation.from_quat(drone_rot).inv().apply(vec_world)
            distance = np.linalg.norm(vec_world)

            label = np.concatenate([vec_local / distance, [distance], cargo_rot])

            self.samples.append(
                {
                    "image": cv2.imread(img_file),
                    "label": torch.tensor(label, dtype=torch.float32),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_file = sample["image"]

        # Return dictionary with all required data
        return {
            "image": utils.preprocess_image(img_file),
            "label": sample["label"],
        }

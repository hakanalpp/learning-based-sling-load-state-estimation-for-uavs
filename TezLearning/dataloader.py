import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import utils


class DroneDataset(Dataset):
    def __init__(
        self,
        images_folder,
        csv_path,
        sequence_length=5,
        step_size=1,  # Control overlap - 1=full overlap, sequence_length=no overlap
    ):
        self.df = pd.read_csv(csv_path, dtype={"frameId": str})
        self.images_folder = images_folder
        self.sequence_length = sequence_length
        self.step_size = step_size

        # Process all samples and store everything we need
        self.samples = []
        for _, row in self.df.iterrows():
            frame_id = row["frameId"]
            img_file = os.path.join(images_folder, f"{frame_id}.jpg")

            # Check if image exists
            if not os.path.exists(img_file):
                continue

            drone_pos = np.array([row["drone_pos_x"], row["drone_pos_y"], row["drone_pos_z"]])
            drone_rot = np.array([row["drone_rot_x"], row["drone_rot_y"], row["drone_rot_z"], row["drone_rot_w"]])
            cargo_pos = np.array([row["cargo_pos_x"], row["cargo_pos_y"], row["cargo_pos_z"]])
            cargo_rot = np.array([row["cargo_rot_x"], row["cargo_rot_y"], row["cargo_rot_z"], row["cargo_rot_w"]])
            cargo_vel_world = np.array([row["cargo_vel_x"], row["cargo_vel_y"], row["cargo_vel_z"]])

            drone_rot_inv = Rotation.from_quat(drone_rot).inv()
            vec_world = cargo_pos - drone_pos
            vec_local = drone_rot_inv.apply(vec_world)
            distance = np.linalg.norm(vec_world)

            if distance < 1e-6:
                continue

            cargo_rot_world = Rotation.from_quat(cargo_rot)
            cargo_rot_local = drone_rot_inv * cargo_rot_world
            cargo_vel_local = drone_rot_inv.apply(cargo_vel_world)

            label = np.concatenate([
                vec_local / distance,
                [distance],
                cargo_rot_local.as_quat(),
                cargo_vel_local,
            ])

            self.samples.append({
                "frame_id": frame_id,
                "image_path": img_file,
                "label": torch.tensor(label, dtype=torch.float32),
            })

        print(f"Total samples loaded: {len(self.samples)}")
        print(f"Step size: {self.step_size} (controls overlap)")
        print(f"Sequences available: {self.__len__()}")

    def __len__(self):
        # Calculate how many sequences we can create with step_size
        max_start_idx = len(self.samples) - self.sequence_length
        if max_start_idx < 0:
            return 0
        
        # Number of sequences = (max_start_idx // step_size) + 1
        return (max_start_idx // self.step_size) + 1

    def __getitem__(self, idx):
        # Convert sequence index to actual starting position based on step_size
        start_idx = idx * self.step_size
        
        # CORRECT: sequence will be [start_idx, start_idx+1, ..., start_idx+seq_len-1]
        # We predict the label for the LAST frame: start_idx + sequence_length - 1
        
        if start_idx + self.sequence_length > len(self.samples):
            raise IndexError(f"Sequence starting at {start_idx} extends beyond dataset")
        
        sequence_images = []
        sequence_frame_ids = []
        
        # Build sequence starting from start_idx
        for i in range(self.sequence_length):
            sample_idx = start_idx + i  # CORRECT: forward indexing
            
            img_path = self.samples[sample_idx]["image_path"]
            img = cv2.imread(img_path)
            processed_img = utils.preprocess_image(img)
            sequence_images.append(processed_img)
            sequence_frame_ids.append(self.samples[sample_idx]["frame_id"])
        
        # Stack into tensor
        image_sequence = torch.stack(sequence_images, dim=0)
        
        # CORRECT: Label from the LAST frame in the sequence
        target_idx = start_idx + self.sequence_length - 1
        label = self.samples[target_idx]["label"]
        
        # # Debug print for first few samples
        # if idx < 3:
        #     print(f"Sequence {idx} (step_size={self.step_size}):")
        #     print(f"  Start index: {start_idx}")
        #     print(f"  Sequence frames: {sequence_frame_ids}")
        #     print(f"  Target frame: {self.samples[target_idx]['frame_id']}")
        #     print(f"  Label shape: {label.shape}")
        
        return {
            "image_sequence": image_sequence,
            "label": label,
            "frame_ids": sequence_frame_ids,  # For debugging
            "target_frame_id": self.samples[target_idx]["frame_id"]  # For debugging
        }
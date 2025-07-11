import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import utils


class StateBasedDroneDataset(Dataset):
    def __init__(
        self,
        images_folder,
        csv_path,
        sequence_length=10,  # Longer sequences for state learning
        augment=False,
        start_from_zero_state=True,  # Whether to simulate starting from unknown state
    ):
        self.df = pd.read_csv(csv_path, dtype={"frameId": str})
        self.images_folder = images_folder
        self.sequence_length = sequence_length
        self.augment = augment
        self.start_from_zero_state = start_from_zero_state

        # First, load and validate all samples
        self.all_samples = []
        for _, row in self.df.iterrows():
            frame_id = row["frameId"]
            img_file = os.path.join(images_folder, f"{frame_id}.jpg")

            if not os.path.exists(img_file):
                continue

            # Process pose data
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
            cargo_vel_world = np.array(
                [row["cargo_vel_x"], row["cargo_vel_y"], row["cargo_vel_z"]]
            )

            # Transform to drone-relative coordinates
            drone_rot_inv = Rotation.from_quat(drone_rot).inv()
            vec_world = cargo_pos - drone_pos
            vec_local = drone_rot_inv.apply(vec_world)
            distance = np.linalg.norm(vec_world)

            if distance < 1e-6:
                continue

            # Relative rotation and velocity
            cargo_rot_world = Rotation.from_quat(cargo_rot)
            cargo_rot_local = drone_rot_inv * cargo_rot_world
            cargo_vel_local = drone_rot_inv.apply(cargo_vel_world)

            # Create label
            label = np.concatenate(
                [
                    vec_local / distance,  # Normalized direction
                    [distance],
                    cargo_rot_local.as_quat(),
                    cargo_vel_local,
                ]
            )

            self.all_samples.append(
                {
                    "frame_id": frame_id,
                    "image_path": img_file,
                    "label": torch.tensor(label, dtype=torch.float32),
                    "timestamp": len(self.all_samples),  # Use index as timestamp
                }
            )

        print(f"Total valid samples: {len(self.all_samples)}")

        # Create sequences with intelligent splitting
        self.sequences = self._create_sequences()

        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")

    def _create_sequences(self):
        """Create sequences with controlled overlap and diversity"""
        sequences = []
        total_frames = len(self.all_samples)

        if total_frames < self.sequence_length:
            print(
                f"Warning: Not enough frames ({total_frames}) for sequence length {self.sequence_length}"
            )
            return sequences

        # Create sequences with calculated stride
        for start_idx in range(0, total_frames - self.sequence_length + 1, self.sequence_length):
            end_idx = start_idx + self.sequence_length

            # Check if sequence is valid (no large time gaps)
            sequence_samples = self.all_samples[start_idx:end_idx]

            # Add sequence
            sequences.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "samples": sequence_samples,
                }
            )

        # Additionally, add some random sequences for diversity
        num_random = min(10, len(sequences) // 4)
        for _ in range(num_random):
            start_idx = np.random.randint(0, total_frames - self.sequence_length + 1)
            end_idx = start_idx + self.sequence_length

            sequences.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "samples": self.all_samples[start_idx:end_idx],
                }
            )

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        samples = sequence_info["samples"]

        # Load images and labels
        images = []
        labels = []
        frame_ids = []

        for sample in samples:
            # Load and preprocess image
            img = cv2.imread(sample["image_path"])
            processed_img = utils.preprocess_image(img)

            # Apply augmentation if enabled
            if self.augment and np.random.rand() > 0.5:
                processed_img = self._augment_image(processed_img)

            images.append(processed_img)
            labels.append(sample["label"])
            frame_ids.append(sample["frame_id"])

        # Stack tensors
        image_sequence = torch.stack(images, dim=0)
        label_sequence = torch.stack(labels, dim=0)

        # For state-based training, we predict all labels in the sequence
        # This helps the model learn to maintain and update state

        output = {
            "image_sequence": image_sequence,
            "label": label_sequence[-1],  # Primary target is last frame
            "label_sequence": label_sequence,  # All labels for auxiliary loss
            "frame_ids": frame_ids,
            "sequence_start": sequence_info["start_idx"],
            "simulate_zero_start": self.start_from_zero_state
            and np.random.rand() > 0.5,
        }

        return output

    def _augment_image(self, image):
        """Apply augmentation to image tensor"""
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)

        # Random contrast adjustment
        if np.random.rand() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean
            image = torch.clamp(image, 0, 1)

        # Random Gaussian noise
        if np.random.rand() > 0.7:
            noise = torch.randn_like(image) * 0.02
            image = image + noise
            image = torch.clamp(image, 0, 1)

        return image

    def get_sequence_statistics(self):
        """Get statistics about the sequences"""
        stats = {
            "total_sequences": len(self.sequences),
            "sequence_length": self.sequence_length,
            "total_frames": len(self.all_samples),
            "sequences_per_epoch": len(self.sequences),
        }

        # Calculate diversity metrics
        start_indices = [seq["start_idx"] for seq in self.sequences]
        stats["unique_start_points"] = len(set(start_indices))
        stats["average_gap_between_starts"] = np.mean(
            np.diff(sorted(set(start_indices)))
        )

        return stats


class MultiRunDroneDataset(Dataset):
    """Dataset that handles multiple runs and creates balanced sequences from each"""

    def __init__(
        self,
        base_folder,
        run_folders,  # List of run folder names
        sequence_length=10,
        augment=False,
    ):
        self.sequence_length = sequence_length
        self.all_sequences = []

        # Load sequences from each run
        for run_folder in run_folders:
            run_path = os.path.join(base_folder, run_folder)
            csv_path = os.path.join(run_path, "cargo_data.csv")

            if not os.path.exists(csv_path):
                print(f"Skipping {run_folder}: CSV not found")
                continue

            print(f"Loading run: {run_folder}")

            # Create dataset for this run
            run_dataset = StateBasedDroneDataset(
                images_folder=run_path,
                csv_path=csv_path,
                sequence_length=sequence_length,
                augment=augment,
            )

            # Add run identifier to sequences
            for seq_idx in range(len(run_dataset)):
                seq_data = run_dataset[seq_idx]
                seq_data["run_id"] = run_folder
                self.all_sequences.append(seq_data)

        print(f"Total sequences from all runs: {len(self.all_sequences)}")

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        return self.all_sequences[idx]

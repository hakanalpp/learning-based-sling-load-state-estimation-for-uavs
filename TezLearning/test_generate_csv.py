import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader import StateBasedDroneDataset
from model import DroneModel

pl.seed_everything(42)


def round_list(arr):
    return [float(f"{x:.2f}") for x in arr]


def collect_and_merge_runs(base_path="/home/alp/noetic_ws/TezLearning/data/images"):
    """
    Collect all runs and merge their CSV data, then shuffle completely
    Same function as in training script
    """
    all_data = []
    run_folders = glob.glob(os.path.join(base_path, "run_*"))
    run_folders.sort()  # Ensure consistent ordering

    print(
        f"Found {len(run_folders)} runs: {[os.path.basename(f) for f in run_folders]}"
    )

    for run_folder in run_folders:
        csv_path = os.path.join(run_folder, "cargo_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add run identifier to track which run each sample came from
            df["run_id"] = os.path.basename(run_folder)

            # CRITICAL: Update frameId to include run folder path
            # The dataloader looks for images using: os.path.join(images_folder, f"{frame_id}.jpg")
            # So we need to prepend the run folder to frameId
            run_name = os.path.basename(run_folder)
            df["frameId"] = df["frameId"].apply(lambda x: f"{run_name}/{x}")

            all_data.append(df)
            print(f"Loaded {len(df)} samples from {run_folder}")
        else:
            print(f"Warning: No CSV found in {run_folder}")

    if not all_data:
        raise ValueError("No valid run data found!")

    # Merge all data
    merged_data = pd.concat(all_data, ignore_index=True)
    print(f"Total merged samples: {len(merged_data)}")

    # Shuffle completely
    merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged_data, base_path


def create_datasets_with_stratified_split():
    """
    Create datasets with proper stratified splitting across runs
    Same function as in training script
    """
    merged_data, base_path = collect_and_merge_runs()

    # Create stratified split to ensure each run is represented in train/val/test
    # Using run_id as stratification key
    train_data, temp_data = train_test_split(
        merged_data, test_size=0.3, stratify=merged_data["run_id"], random_state=42
    )

    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=temp_data["run_id"], random_state=42
    )

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print("Run distribution in splits:")
    for split_name, split_data in [
        ("Train", train_data),
        ("Val", val_data),
        ("Test", test_data),
    ]:
        run_counts = split_data["run_id"].value_counts()
        print(f"  {split_name}: {dict(run_counts)}")

    # Create datasets using the split data
    train_dataset = StateBasedDroneDataset(
        images_folder=base_path,
        csv_data=train_data,  # Pass dataframe directly instead of CSV path
    )

    val_dataset = StateBasedDroneDataset(images_folder=base_path, csv_data=val_data)

    test_dataset = StateBasedDroneDataset(images_folder=base_path, csv_data=test_data)

    return train_dataset, val_dataset, test_dataset, test_data


class TestCsvGenerate:
    def __init__(self, model_path, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DroneModel.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.data_path = Path(data_path)

        # Fixed camera parameters
        self.fx = self.fy = 112
        self.cx = self.cy = 112

    def create_test_dataset(self):
        """Create test dataset using the same logic as training script"""
        _, _, test_dataset, test_data = create_datasets_with_stratified_split()

        # Store the test data DataFrame for lookup
        self.test_df = test_data.copy()

        return test_dataset

    def world_to_camera_coords(self, world_vector, camera_rot):
        """Convert world coordinates to camera coordinates"""
        cam_R = Rotation.from_quat(camera_rot)
        P_cam = cam_R.as_matrix().T @ world_vector
        return P_cam

    def camera_to_pixel_coords(self, P_cam):
        """Convert camera coordinates to pixel coordinates"""
        if P_cam[2] <= 0:  # Behind camera
            return None, None

        x_norm = P_cam[0] / P_cam[2]
        y_norm = -(P_cam[1] / P_cam[2])

        u = int(self.fx * x_norm + self.cx)
        v = int(self.fy * y_norm + self.cy)

        return u, v

    def evaluate_test_set(self):
        test_dataset = self.create_test_dataset()
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        )

        results = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating test set"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                frame_ids = batch["frame_id"]

                # Get model predictions
                predictions = self.model(images)

                # Process each sample in the batch
                batch_size = images.shape[0]
                for i in range(batch_size):
                    frame_id = frame_ids[i]  # This is now a string like "run_0/12345"

                    # Get ground truth from merged CSV using string frameId
                    row = self.test_df[self.test_df["frameId"] == frame_id].iloc[0]

                    # Ground truth positions and rotations
                    drone_pos = np.array(
                        [row["drone_pos_x"], row["drone_pos_y"], row["drone_pos_z"]]
                    )
                    cargo_pos = np.array(
                        [row["cargo_pos_x"], row["cargo_pos_y"], row["cargo_pos_z"]]
                    )
                    camera_rot = np.array(
                        [
                            row["camera_rot_x"],
                            row["camera_rot_y"],
                            row["camera_rot_z"],
                            row["camera_rot_w"],
                        ]
                    )

                    # Ground truth direction vector (in world coordinates)
                    gt_world_vector = cargo_pos - drone_pos
                    gt_cam_vector = self.world_to_camera_coords(
                        gt_world_vector, camera_rot
                    )
                    gt_u, gt_v = self.camera_to_pixel_coords(gt_cam_vector)

                    # Model prediction (8-dimensional output: 3 direction + 1 distance + 4 rotation)
                    pred_tensor = predictions[i].cpu().numpy()
                    gt_tensor = labels[i].cpu().numpy()

                    # Extract predicted components
                    pred_direction = pred_tensor[:3]  # Camera coordinates
                    pred_distance = pred_tensor[3]
                    pred_rotation = pred_tensor[4:8]

                    # Scale direction by distance to get predicted camera vector
                    pred_cam_vector = pred_direction * pred_distance
                    pred_u, pred_v = self.camera_to_pixel_coords(pred_cam_vector)

                    # Convert camera coordinates back to world coordinates
                    cam_R = Rotation.from_quat(camera_rot)
                    pred_world_vector = cam_R.as_matrix() @ pred_cam_vector

                    # Calculate losses
                    # Direction loss (using normalized vectors)
                    pred_dir_norm = pred_direction / (
                        np.linalg.norm(pred_direction) + 1e-8
                    )
                    gt_dir_norm = gt_tensor[:3] / (np.linalg.norm(gt_tensor[:3]) + 1e-8)
                    dir_loss = 1 - np.dot(pred_dir_norm, gt_dir_norm)

                    # Distance loss
                    dist_loss = (pred_distance - gt_tensor[3]) ** 2

                    # Rotation loss (quaternion similarity)
                    pred_quat_norm = pred_rotation / (
                        np.linalg.norm(pred_rotation) + 1e-8
                    )
                    gt_quat_norm = gt_tensor[4:8] / (
                        np.linalg.norm(gt_tensor[4:8]) + 1e-8
                    )
                    rot_loss = 1 - abs(np.dot(pred_quat_norm, gt_quat_norm))

                    # Pixel error
                    pixel_error = None
                    if gt_u is not None and pred_u is not None:
                        pixel_error = np.sqrt(
                            (gt_u - pred_u) ** 2 + (gt_v - pred_v) ** 2
                        )

                    # Store results
                    result = {
                        "frameId": frame_id,  # Keep as string
                        "gt_u": gt_u if gt_u is not None else -1,
                        "gt_v": gt_v if gt_v is not None else -1,
                        "pred_u": pred_u if pred_u is not None else -1,
                        "pred_v": pred_v if pred_v is not None else -1,
                        "gt_world_vector": round_list(gt_world_vector.tolist()),
                        "pred_world_vector": round_list(pred_world_vector.tolist()),
                        "gt_cam_vector": round_list(gt_cam_vector.tolist()),
                        "pred_cam_vector": round_list(pred_cam_vector.tolist()),
                        "dir_loss": float(f"{dir_loss:.4f}"),
                        "dist_loss": float(f"{dist_loss:.4f}"),
                        "rot_loss": float(f"{rot_loss:.4f}"),
                        "pixel_error": float(f"{pixel_error:.2f}")
                        if pixel_error is not None
                        else -1,
                        "gt_direction": round_list(gt_tensor[:3].tolist()),
                        "pred_direction": round_list(pred_direction.tolist()),
                        "gt_distance": float(f"{gt_tensor[3]:.2f}"),
                        "pred_distance": float(f"{pred_distance:.2f}"),
                        "gt_rotation": round_list(gt_tensor[4:8].tolist()),
                        "pred_rotation": round_list(pred_rotation.tolist()),
                        "run_id": row["run_id"],  # Add run_id for analysis
                    }

                    results.append(result)

        return results

    def save_results_to_csv(self, results, output_path="test_results.csv"):
        """Save evaluation results to CSV"""
        if not results:
            print("No results to save!")
            return

        fieldnames = list(results[0].keys())

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {output_path}")

        # Print summary statistics
        valid_results = [r for r in results if r["pixel_error"] != -1]
        if valid_results:
            avg_pixel_error = np.mean([r["pixel_error"] for r in valid_results])
            avg_dir_loss = np.mean([r["dir_loss"] for r in results])
            avg_dist_loss = np.mean([r["dist_loss"] for r in results])
            avg_rot_loss = np.mean([r["rot_loss"] for r in results])

            print(f"\nTest Set Summary:")
            print(f"Average Pixel Error: {avg_pixel_error:.2f}")
            print(f"Average Direction Loss: {avg_dir_loss:.4f}")
            print(f"Average Distance Loss: {avg_dist_loss:.4f}")
            print(f"Average Rotation Loss: {avg_rot_loss:.4f}")
            print(f"Valid predictions: {len(valid_results)}/{len(results)}")

            # Print per-run statistics
            print(f"\nPer-Run Statistics:")
            run_stats = {}
            for result in valid_results:
                run_id = result["run_id"]
                if run_id not in run_stats:
                    run_stats[run_id] = []
                run_stats[run_id].append(result["pixel_error"])

            for run_id, errors in run_stats.items():
                avg_error = np.mean(errors)
                print(f"  {run_id}: {avg_error:.2f} px (n={len(errors)})")


def main():
    evaluator = TestCsvGenerate(
        model_path="/home/alp/noetic_ws/TezLearning/checkpoints/cargo_model-epoch=11-val_loss=0.0362.ckpt",
        data_path="/home/alp/noetic_ws/TezLearning/data/images",
    )

    print("Starting test set evaluation...")
    results = evaluator.evaluate_test_set()

    print("Saving results...")
    evaluator.save_results_to_csv(results, "test_results.csv")

    print("Evaluation complete!")


if __name__ == "__main__":
    main()

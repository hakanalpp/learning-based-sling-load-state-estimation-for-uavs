import ast
import glob
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def collect_and_merge_original_data(base_path="data/images"):
    """
    Collect and merge original CSV data from all runs
    """
    all_data = []
    run_folders = glob.glob(os.path.join(base_path, "run_*"))
    run_folders.sort()

    for run_folder in run_folders:
        csv_path = os.path.join(run_folder, "cargo_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Update frameId to match the format used in test results
            run_name = os.path.basename(run_folder)
            df["frameId"] = df["frameId"].apply(lambda x: f"{run_name}/{x}")
            all_data.append(df)

    merged_original = pd.concat(all_data, ignore_index=True)
    return merged_original


class PredictionVisualizer:
    def __init__(self, images_folder, csv_path, test_results_path, output_folder):
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)

        self.df = collect_and_merge_original_data(csv_path)

        self.test_results = pd.read_csv(test_results_path)

        # OPTIMIZATION 1: Create frameId lookup dictionary for O(1) access
        print("Creating frameId lookup...")
        self.gt_lookup = {}
        for _, row in self.df.iterrows():
            self.gt_lookup[row["frameId"]] = row
        print(f"Created lookup for {len(self.gt_lookup)} frames")

        # OPTIMIZATION 2: Pre-parse all string columns to avoid repeated ast.literal_eval
        print("Pre-parsing test results...")
        self.test_results["pred_direction_parsed"] = self.test_results[
            "pred_direction"
        ].apply(ast.literal_eval)
        self.test_results["pred_rotation_parsed"] = self.test_results[
            "pred_rotation"
        ].apply(ast.literal_eval)

    def camera_to_pixel_coords(self, P_cam):
        x_norm = P_cam[0] / P_cam[2]
        y_norm = -(P_cam[1] / P_cam[2])
        u = int(112 * x_norm + 112)
        v = int(112 * y_norm + 112)
        return u, v

    def project_3d_box(
        self, center_cam, box_size=(1.0, 1.0, 1.0), rotation_matrix=None
    ):
        w, h, d = box_size
        vertices = np.array(
            [
                [-w / 2, -h / 2, -d / 2],
                [w / 2, -h / 2, -d / 2],
                [w / 2, h / 2, -d / 2],
                [-w / 2, h / 2, -d / 2],
                [-w / 2, -h / 2, d / 2],
                [w / 2, -h / 2, d / 2],
                [w / 2, h / 2, d / 2],
                [-w / 2, h / 2, d / 2],
            ]
        )

        if rotation_matrix is not None:
            vertices = vertices @ rotation_matrix.T

        world_vertices = vertices + center_cam

        projected_points = []
        for vertex in world_vertices:
            u, v = self.camera_to_pixel_coords(vertex)
            projected_points.append((u, v))

        return projected_points

    def draw_3d_box(self, img, projected_points, color):
        face4_edges = [(2, 3), (3, 7), (7, 6), (6, 2)]
        for start_idx, end_idx in face4_edges:
            start_point = projected_points[start_idx]
            end_point = projected_points[end_idx]
            cv2.line(img, start_point, end_point, color, 2)

    def visualize_frame(self, test_idx):
        test_row = self.test_results.iloc[test_idx]

        # Get the frameId from test results (now includes run prefix like "run_0/12345")
        frame_id = test_row["frameId"]

        # Find the corresponding ground truth row using frameId
        gt_rows = self.df[self.df["frameId"] == frame_id]
        if gt_rows.empty:
            print(f"Warning: No ground truth found for frameId {frame_id}")
            return

        gt_row = gt_rows.iloc[0]

        # Construct image path based on frameId format
        if "/" in frame_id:
            # Multi-run format: "run_0/12345" -> "data/images/run_0/12345.jpg"
            run_name, numeric_id = frame_id.split("/")
            img_path = self.images_folder / run_name / f"{numeric_id}.jpg"
        else:
            # Single run format: "12345" -> "data/images/12345.jpg"
            img_path = self.images_folder / f"{frame_id}.jpg"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            return

        # Ground truth
        drone_pos = np.array(
            [gt_row["drone_pos_x"], gt_row["drone_pos_y"], gt_row["drone_pos_z"]]
        )
        cargo_pos = np.array(
            [gt_row["cargo_pos_x"], gt_row["cargo_pos_y"], gt_row["cargo_pos_z"]]
        )
        camera_rot = np.array(
            [
                gt_row["camera_rot_x"],
                gt_row["camera_rot_y"],
                gt_row["camera_rot_z"],
                gt_row["camera_rot_w"],
            ]
        )

        cargo_rot = np.array(
            [
                gt_row["cargo_rot_x"],
                gt_row["cargo_rot_y"],
                gt_row["cargo_rot_z"],
                gt_row["cargo_rot_w"],
            ]
        )

        cam_R = Rotation.from_quat(camera_rot).as_matrix()
        cargo_R = Rotation.from_quat(cargo_rot).as_matrix()

        gt_world_vector = cargo_pos - drone_pos
        gt_cam_vector = cam_R.T @ gt_world_vector

        cargo_R_cam = cam_R.T @ cargo_R

        # Prediction
        pred_direction = np.array(ast.literal_eval(test_row["pred_direction"]))
        pred_distance = test_row["pred_distance"]

        # Ground truth box with rotation
        gt_box_points = self.project_3d_box(
            gt_cam_vector, (0.33, 0.33, 0.33), cargo_R_cam
        )
        self.draw_3d_box(img, gt_box_points, (0, 255, 0))  # Blue

        # Prediction visualization
        cam_R = Rotation.from_quat(camera_rot)

        # P_cam = pred_direction * pred_distance

        # x_norm = P_cam[0] / P_cam[2]
        # y_norm = -(P_cam[1] / P_cam[2])

        # fx = fy = 112
        # cx = cy = 112

        # u = int(fx * x_norm + cx)
        # v = int(fy * y_norm + cy)

        pred_box_points = self.project_3d_box(
            pred_direction, (0.33, 0.33, 0.33), cargo_R_cam
        )
        self.draw_3d_box(img, pred_box_points, (255, 0, 0))  # Green

        # # Draw prediction point for debugging
        # cv2.circle(img, (u, v), 5, (0, 255, 0), -1)

        run_name, numeric_id = frame_id.split("/")
        output_filename = f"pred_{run_name}_{numeric_id}.jpg"

        cv2.imwrite(str(self.output_folder / output_filename), img)

    def visualize_all(self, max_samples=None):
        """
        Visualize test results with optional limit for faster testing

        Args:
            max_samples: If provided, only visualize first N samples for testing
        """
        total_samples = len(self.test_results)
        if max_samples is not None:
            total_samples = min(max_samples, total_samples)
            print(
                f"Visualizing first {total_samples} of {len(self.test_results)} test results..."
            )
        else:
            print(f"Visualizing all {total_samples} test results...")

        for test_idx in tqdm(range(total_samples)):
            self.visualize_frame(test_idx)


def main():
    if os.path.exists("prediction_visualization"):
        shutil.rmtree("prediction_visualization")

    visualizer = PredictionVisualizer(
        images_folder="data/images",
        csv_path="data/images",
        test_results_path="test_results.csv",
        output_folder="prediction_visualization",
    )

    visualizer.visualize_all(max_samples=18000)


if __name__ == "__main__":
    main()

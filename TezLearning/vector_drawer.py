import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class DirectionVectorVisualizer:
    def __init__(self, images_folder, csv_path, output_folder):
        self.images_folder = Path(images_folder)
        self.csv_path = csv_path
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)

        self.df = pd.read_csv(csv_path)

        width, height, fov_rad = 224, 224, np.deg2rad(90)
        focal_length = width / (2 * np.tan(fov_rad / 2))

        self.K = np.array(
            [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]]
        )

    def visualize_frame(self, frame_id):
        row = self.df[self.df["frameId"] == int(frame_id)].iloc[0]
        img = cv2.imread(str(self.images_folder / f"{frame_id}.jpg")).copy()

        drone_pos = np.array(
            [row["drone_pos_x"], row["drone_pos_y"], row["drone_pos_z"]]
        )
        cargo_pos = np.array(
            [row["cargo_pos_x"], row["cargo_pos_y"], row["cargo_pos_z"]]
        )

        camera_rot = np.array(
            [row["camera_rot_x"], row["camera_rot_y"], row["camera_rot_z"]]
        )
        R = Rotation.from_euler(
            "XYZ", camera_rot, degrees=True
        ).as_matrix()  # Add degrees=True if needed
        P_cam = R @ (cargo_pos - drone_pos)

        # if P_cam[2] <= 0:
        # print("negatif knk")
        # print(P_cam, R)
        # return img

        x_norm = P_cam[0] / P_cam[2]
        y_norm = -(P_cam[1] / P_cam[2])

        fx = fy = 112
        cx = cy = 112

        u = int(fx * x_norm + cx)
        v = int(fy * y_norm + cy)
        np.set_printoptions(precision=2, suppress=True)

        print(frame_id, u, v, P_cam, cargo_pos - drone_pos, camera_rot)

        if 0 > u > 224 and 0 > v > 224:
            print("u v bozuk knk")
            return img

        # print("güzel knk")
        cv2.circle(img, (int(u), int(v)), 3, (255, 0, 0), -1)
        output_path = self.output_folder / f"vis_{frame_id}.jpg"
        cv2.imwrite(str(output_path), img)

        return img

    def visualize_all(self):
        frame_ids = self.df["frameId"].values

        for frame_id in tqdm(frame_ids):
            if os.path.exists(f"data/images/{frame_id}.jpg"):
                self.visualize_frame(frame_id)


def main():
    if os.path.exists("data_visualization"):
        shutil.rmtree("data_visualization")
    visualizer = DirectionVectorVisualizer(
        images_folder="data/images",
        csv_path="data/images/cargo_data.csv",
        output_folder="data_visualization",
    )

    visualizer.visualize_all()


if __name__ == "__main__":
    main()

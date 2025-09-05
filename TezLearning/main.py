import time
from typing import List

import numpy as np
import torch

import utils
from model import DroneModel
from tcp import TCPHandler


class ModelProcessor:
    def __init__(
        self,
        model_path: str = "weights.ckpt",
        image_dimensions: tuple = (224, 224),
        image_host: str = "0.0.0.0",
        image_port: int = 10001,
        data_host: str = "127.0.0.1",
        data_port: int = 9997,
    ):
        self.tcp_handler = TCPHandler(
            image_host=image_host,
            image_port=image_port,
            data_host=data_host,
            data_port=data_port,
            image_dimensions=image_dimensions,
        )
        self.model_path = model_path

        self.model = self._load_model()
        self.device = next(self.model.parameters()).device 

        self.frame_count = 0

    def _load_model(self):
        """Load the model from checkpoint"""
        try:
            device = "cuda"
            model = DroneModel.load_from_checkpoint(checkpoint_path=self.model_path)

            model.eval()
            model.to(device)
            print(f"Model loaded successfully from {self.model_path} on {device}")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _evaluate(self, img: np.ndarray):
        """Run the model on new image"""
        try:
            img_tensor = utils.preprocess_image(img)
            
            # Single image inference
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                predictions = self.model(img_tensor)

            pred_numpy = predictions[0].cpu().numpy()
            self.frame_count += 1

            return pred_numpy.tolist()

        except Exception as e:
            print(f"Error evaluating model: {e}")
            return [0.0] * 8

    def _evaluate_and_send(
        self,
        img: np.ndarray,
        label: List[float],
    ) -> None:
        """Process image and send predictions"""
        float_values = self._evaluate(img)
        # Calculate predicted position with safety checks
        pred_direction = np.array([float_values[0], float_values[1], float_values[2]])
        pred_norm = np.linalg.norm(pred_direction)
        if pred_norm > 1e-6:
            pred_direction = pred_direction / pred_norm
        else:
            pred_direction = np.array([0, 0, 1])  # Default forward
            
        pred_distance = float_values[3]
        # Clamp distance to reasonable range
        pred_distance = np.clip(pred_distance, 0.1, 100.0)
        pred_position_camera_frame = pred_direction * pred_distance
        
        # Calculate actual position with safety checks
        actual_direction = np.array([label[0], label[1], label[2]])
        actual_norm = np.linalg.norm(actual_direction)
        if actual_norm > 1e-6:
            actual_direction = actual_direction / actual_norm
        else:
            actual_direction = np.array([0, 0, 1])  # Default forward
            
        actual_distance = label[3]
        actual_distance = np.clip(actual_distance, 0.1, 100.0)
        actual_position_camera_frame = actual_direction * actual_distance
        
        # Calculate position difference (in camera frame)
        position_diff = pred_position_camera_frame - actual_position_camera_frame
        position_error_magnitude = np.linalg.norm(position_diff)
        
        # Calculate angle error between direction vectors (in degrees)
        dot_product = np.clip(np.dot(pred_direction, actual_direction), -1.0, 1.0)
        angle_error_deg = np.degrees(np.arccos(dot_product))
        
        # Calculate rotation error
        pred_quat = np.array([float_values[4], float_values[5], float_values[6], float_values[7]])
        actual_quat = np.array([label[4], label[5], label[6], label[7]])
        
        # Normalize quaternions
        pred_quat = pred_quat / np.linalg.norm(pred_quat)
        actual_quat = actual_quat / np.linalg.norm(actual_quat)
        
        # Calculate rotation error in degrees
        dot_quat = abs(np.dot(pred_quat, actual_quat))
        dot_quat = np.clip(dot_quat, 0.0, 1.0)
        rotation_error_deg = 2 * np.degrees(np.arccos(dot_quat))
        
        print(f"PosErr:{position_error_magnitude:.2f} DirErr:{angle_error_deg:.1f}° RotErr:{rotation_error_deg:.1f}°")
        
        self.tcp_handler.send_floats(float_values)

    def run(self) -> None:
        """Main execution loop"""
        try:
            print("Starting model processor")
            self.tcp_handler.start_receiver(self._evaluate_and_send)
            print("Model processor running. Press Ctrl+C to stop.")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            self.tcp_handler.stop()
            print("Processor terminated.")


if __name__ == "__main__":
    processor = ModelProcessor(model_path="/home/alp/noetic_ws/TezLearning/checkpoints/cargo_model-epoch=11-val_loss=0.0362.ckpt")
    processor.run()
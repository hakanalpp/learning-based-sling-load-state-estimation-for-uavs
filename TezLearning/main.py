import os
import time
from typing import List

import numpy as np
import torch

import utils
from model import CargoPoseModel
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
        os.makedirs("recv_frames_raw", exist_ok=True)
        self.tcp_handler = TCPHandler(
            image_host=image_host,
            image_port=image_port,
            data_host=data_host,
            data_port=data_port,
            image_dimensions=image_dimensions,
        )
        self.model_path = model_path
        self.model = self._load_model()
        self.total_processing_times = []

    def _load_model(self):
        try:
            model = CargoPoseModel.load_from_checkpoint(checkpoint_path=self.model_path)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict_and_send(
        self,
        img: np.ndarray,
        position: List[float],
        rotation: List[float],
    ) -> None:
        start_time = time.time()

        float_values = self.predict_from_frame(img)
        self.tcp_handler.send_floats(float_values, position, rotation)

        self.total_processing_times.append((time.time() - start_time) * 1000)

    def predict_from_frame(self, img: np.ndarray) -> List[float]:
        try:
            img_tensor = utils.preprocess_image(img).unsqueeze(0)
            device = next(self.model.parameters()).device
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                model_output = self.model(img_tensor)

            return model_output.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"Error processing image with model: {e}")
            return [1.0] * 8

    def run(self) -> None:
        if self.model is None:
            print("Model not loaded. Exiting.")
            return
        try:
            self.tcp_handler.start_receiver(self.predict_and_send)
            print("Model processor running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.tcp_handler.stop()
            print("Processor terminated.")

            if self.total_processing_times:
                min_time = min(self.total_processing_times)
                max_time = max(self.total_processing_times)
                avg_time = sum(self.total_processing_times) / len(
                    self.total_processing_times
                )
                print(
                    f"Final stats - Min: {min_time:.2f} ms, Max: {max_time:.2f} ms, Avg: {avg_time:.2f} ms, Total frames: {len(self.total_processing_times)}"
                )


if __name__ == "__main__":
    processor = ModelProcessor()
    processor.run()

import time
from collections import deque
from typing import List

import numpy as np
import torch

import utils
from model import Model
from tcp import TCPHandler


class ModelProcessor:
    def __init__(
        self,
        model_path: str = "weights.ckpt",
        sequence_length: int = 5,
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
        self.sequence_length = sequence_length
        self.model = self._load_model()
        self.device = next(self.model.parameters()).device

        self.total_processing_times = []

        self.frame_buffer = deque(maxlen=sequence_length)
        self.label_buffer = deque(maxlen=sequence_length)

        self.accumulated_metrics = {}
        self.frame_count = 0

    def _load_model(self):
        """Load the PyTorch Lightning model from checkpoint"""
        try:
            model = Model.load_from_checkpoint(
                checkpoint_path=self.model_path,
                λ_rot=0.8,
                λ_dir=0.5,
                λ_dist=0.1,
                λ_pos=1.5,
                λ_vel=0.3,
                sequence_length=self.sequence_length,
            )
            model.eval()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Using device: {device}")
            print(f"Sequence length: {self.sequence_length}")

            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _evaluate(self, img: np.ndarray, label: List[float]):
        """
        Run the model's test_step on a sequence of images with ground truth values.
        Returns predictions or zeros if buffer not filled yet.
        """
        try:
            img_tensor = utils.preprocess_image(img)
            self.frame_buffer.append(img_tensor)
            self.label_buffer.append(label)

            if len(self.frame_buffer) < self.sequence_length:
                return [0.0] * 11

            sequence_tensor = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0)
            sequence_tensor = sequence_tensor.to(self.device)

            gt_tensor = (
                torch.tensor(label, dtype=torch.float32).unsqueeze(0).to(self.device)
            )

            batch = {"image_sequence": sequence_tensor, "label": gt_tensor}
            with torch.no_grad():
                pred = self.model(sequence_tensor)
                test_results = self.model.test_step(batch, 0)

            print(f"\nFrame {self.frame_count + 1}:")
            utils.print_test_metrics(test_results)

            utils.check_for_nan_values(test_results, pred, gt_tensor, label)

            utils.accumulate_metrics(self.accumulated_metrics, test_results)
            self.frame_count += 1

            return pred.squeeze().cpu().numpy().tolist()

        except Exception as e:
            print(f"Error evaluating model: {e}")
            import traceback

            traceback.print_exc()
            return [0.0] * 11

    def _evaluate_and_send(
        self,
        img: np.ndarray,
        position: List[float],
        rotation: List[float],
        label: List[float],
    ) -> None:
        """Process image and send predictions"""
        start_time = time.time()

        # Get predictions (will be zeros if buffer not filled)
        float_values = self._evaluate(img, label)

        # Send the predictions (zeros or actual predictions)
        self.tcp_handler.send_floats(float_values, position, rotation)

        self.total_processing_times.append((time.time() - start_time) * 1000)

        # Print buffer status for first few frames
        if len(self.frame_buffer) < self.sequence_length:
            print(
                f"Filling buffer: {len(self.frame_buffer)}/{self.sequence_length} - Sending zeros"
            )

    def run(self) -> None:
        """Main execution loop"""
        if self.model is None:
            print("Model not loaded. Exiting.")
            return
        try:
            print(
                f"Starting model processor with sequence length {self.sequence_length}"
            )
            print(
                f"First {self.sequence_length} frames will send zeros while filling buffer..."
            )

            self.tcp_handler.start_receiver(self._evaluate_and_send)
            print("Model processor running. Press Ctrl+C to stop.")
            print("Evaluating frames... (metrics will be shown when stopped)")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            self.tcp_handler.stop()
            print("\nProcessor terminated.")

            # Print final averaged metrics using utils function
            utils.print_final_metrics(self.frame_count, self.accumulated_metrics)

            # Print processing time stats
            if self.total_processing_times:
                min_time = min(self.total_processing_times)
                max_time = max(self.total_processing_times)
                avg_time = sum(self.total_processing_times) / len(
                    self.total_processing_times
                )
                print(f"\nProcessing time stats:")
                print(f"  Min: {min_time:.2f} ms")
                print(f"  Max: {max_time:.2f} ms")
                print(f"  Avg: {avg_time:.2f} ms")


if __name__ == "__main__":
    processor = ModelProcessor()
    processor.run()

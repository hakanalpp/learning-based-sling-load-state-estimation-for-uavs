import csv
import os
import time
from collections import deque
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils
from model import StateBasedModel
from tcp import TCPHandler


class StateBasedModelProcessor:
    def __init__(
        self,
        model_path: str = "state_based_model_final.ckpt",
        sequence_length: int = 10,
        image_dimensions: tuple = (224, 224),
        image_host: str = "0.0.0.0",
        image_port: int = 10001,
        data_host: str = "127.0.0.1",
        data_port: int = 9997,
        warm_up_frames: int = 5,
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
        self.warm_up_frames = warm_up_frames
        
        # Initialize CSV logging
        self._init_csv_logging()
        
        self.model = self._load_model()
        if self.model is not None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device("cpu")  # Fallback device

        self.current_state = None
        self.frame_buffer = deque(maxlen=sequence_length)
        self.prediction_buffer = deque(maxlen=10)
        self.uncertainty_buffer = deque(maxlen=5)
        
        self.total_processing_times = []
        self.accumulated_metrics = {}
        self.frame_count = 0
        self.frames_since_reset = 0
        self.start_time = time.time()
        
        self.last_valid_distance = None
        self.confidence_threshold = 0.8
        self.outlier_threshold = 25.0

    def _init_csv_logging(self):
        """Initialize CSV file for error logging"""
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"results/prediction_errors_{timestamp}.csv"
        
        headers = [
            'frame_id', 'timestamp', 'relative_time',
            'pred_dir_x', 'pred_dir_y', 'pred_dir_z',
            'gt_dir_x', 'gt_dir_y', 'gt_dir_z',
            'pred_distance', 'gt_distance',
            'pred_rot_x', 'pred_rot_y', 'pred_rot_z', 'pred_rot_w',
            'gt_rot_x', 'gt_rot_y', 'gt_rot_z', 'gt_rot_w',
            'pred_vel_x', 'pred_vel_y', 'pred_vel_z',
            'gt_vel_x', 'gt_vel_y', 'gt_vel_z',
            'direction_error_degrees', 'distance_error_abs', 'distance_error_rel',
            'rotation_error_degrees', 'velocity_error_magnitude',
            'position_error_magnitude', 'uncertainty', 'processing_time_ms'
        ]
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    def _save_to_csv(self, metrics_data):
        """Save metrics data to CSV"""
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(metrics_data)

    def _load_model(self):
        """Load the PyTorch Lightning model from checkpoint with proper device handling"""
        try:
            # Force CPU loading if CUDA has issues
            device = "cpu"
            if torch.cuda.is_available():
                try:
                    # Test CUDA
                    torch.cuda.current_device()
                    device = "cuda"
                except:
                    print("CUDA available but not working properly, using CPU")
                    device = "cpu"
            
            # Load model with explicit device mapping
            if device == "cpu":
                model = StateBasedModel.load_from_checkpoint(
                    checkpoint_path=self.model_path,
                    map_location=torch.device('cpu')
                )
            else:
                model = StateBasedModel.load_from_checkpoint(
                    checkpoint_path=self.model_path
                )
            
            model.eval()
            model.to(device)
            print(f"State-based model loaded successfully from {self.model_path}")
            print(f"Using device: {device}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing without model - will return zero predictions")
            return None

    def _calculate_errors(self, pred_numpy, label):
        """Calculate comprehensive errors between prediction and ground truth"""
        try:
            # Extract components
            pred_dir = pred_numpy[:3]
            pred_dist = pred_numpy[3]
            pred_rot = pred_numpy[4:8]
            pred_vel = pred_numpy[8:11]
            
            gt_dir = np.array(label[:3])
            gt_dist = label[3]
            gt_rot = np.array(label[4:8])
            gt_vel = np.array(label[8:11])
            
            # Direction error (angular)
            pred_dir_norm = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
            gt_dir_norm = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)
            cos_sim = np.clip(np.dot(pred_dir_norm, gt_dir_norm), -1.0, 1.0)
            direction_error_degrees = np.degrees(np.arccos(np.abs(cos_sim)))
            
            # Distance errors
            distance_error_abs = abs(pred_dist - gt_dist)
            distance_error_rel = distance_error_abs / max(abs(gt_dist), 1e-6) * 100
            
            # Rotation error (quaternion angular difference)
            pred_rot_norm = pred_rot / (np.linalg.norm(pred_rot) + 1e-8)
            gt_rot_norm = gt_rot / (np.linalg.norm(gt_rot) + 1e-8)
            dot_product = np.abs(np.dot(pred_rot_norm, gt_rot_norm))
            dot_product = np.clip(dot_product, 0, 1)
            rotation_error_degrees = 2 * np.degrees(np.arccos(dot_product))
            
            # Velocity errors
            velocity_error = pred_vel - gt_vel
            velocity_error_magnitude = np.linalg.norm(velocity_error)
            
            # Position errors
            pred_pos = pred_dir_norm * pred_dist
            gt_pos = gt_dir_norm * gt_dist
            position_error = pred_pos - gt_pos
            position_error_magnitude = np.linalg.norm(position_error)
            
            return {
                'direction_error_degrees': direction_error_degrees,
                'distance_error_abs': distance_error_abs,
                'distance_error_rel': distance_error_rel,
                'rotation_error_degrees': rotation_error_degrees,
                'velocity_error_magnitude': velocity_error_magnitude,
                'position_error_magnitude': position_error_magnitude,
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'direction_error_degrees': 0.0,
                'distance_error_abs': 0.0,
                'distance_error_rel': 0.0,
                'rotation_error_degrees': 0.0,
                'velocity_error_magnitude': 0.0,
                'position_error_magnitude': 0.0,
            }

    def _should_reset_state(self, prediction, uncertainty):
        pred_distance = prediction[3]
        
        if pred_distance < 0.1 or pred_distance > 50.0:
            return True
            
        if uncertainty > 5.0:
            return True
            
        if self.last_valid_distance is not None:
            distance_change = abs(pred_distance - self.last_valid_distance)
            if distance_change > 5.0:
                return True
                
        return False

    def _smooth_predictions(self, prediction):
        self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) < 3:
            return prediction
        
        predictions_array = np.array(list(self.prediction_buffer))
        
        if len(self.prediction_buffer) >= 5:
            directions = predictions_array[-5:, :3]
            median_direction = np.median(directions, axis=0)
            
            angle_errors = []
            for d in directions:
                d_norm = d / (np.linalg.norm(d) + 1e-8)
                median_norm = median_direction / (np.linalg.norm(median_direction) + 1e-8)
                cos_sim = np.clip(np.dot(d_norm, median_norm), -1.0, 1.0)
                angle_errors.append(np.degrees(np.arccos(cos_sim)))
            
            if max(angle_errors) > self.outlier_threshold:
                valid_indices = [i for i, err in enumerate(angle_errors) if err < self.outlier_threshold]
                if valid_indices:
                    valid_directions = directions[valid_indices]
                    prediction[:3] = np.mean(valid_directions, axis=0)
        
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])[-len(self.prediction_buffer):]
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(prediction)
        for i, (weight, pred) in enumerate(zip(weights, self.prediction_buffer)):
            smoothed += weight * pred
            
        # Normalize direction and quaternion
        if np.linalg.norm(smoothed[:3]) > 1e-8:
            smoothed[:3] = smoothed[:3] / np.linalg.norm(smoothed[:3])
        if np.linalg.norm(smoothed[4:8]) > 1e-8:
            smoothed[4:8] = smoothed[4:8] / np.linalg.norm(smoothed[4:8])
        
        return smoothed

    def _evaluate(self, img: np.ndarray, label: List[float]):
        """
        Run the state-based model on new image with ground truth values.
        """
        processing_start = time.time()
        
        try:
            # If model failed to load, return zeros
            if self.model is None:
                print("Model not loaded, returning zero predictions")
                return [0.0] * 11
            
            img_tensor = utils.preprocess_image(img)
            self.frame_buffer.append(img_tensor)
            
            # During warm-up, just collect frames
            if len(self.frame_buffer) < self.warm_up_frames:
                print(f"Warming up: {len(self.frame_buffer)}/{self.warm_up_frames} frames")
                return [0.0] * 11
            
            # Prepare input
            if len(self.frame_buffer) < self.sequence_length:
                frames = list(self.frame_buffer)
                while len(frames) < self.sequence_length:
                    frames.insert(0, frames[0])
                sequence_tensor = torch.stack(frames, dim=0).unsqueeze(0)
            else:
                sequence_tensor = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0)
            
            sequence_tensor = sequence_tensor.to(self.device)
            
            # Run model with current state
            with torch.no_grad():
                predictions, states, uncertainties = self.model(
                    sequence_tensor, 
                    initial_state=self.current_state,
                    return_all_states=True
                )
                
                # Update current state
                self.current_state = states[-1].detach()
                
                # Get prediction and uncertainty
                if len(predictions.shape) == 3:
                    pred = predictions[0, -1]
                    uncertainty = uncertainties[0, -1]
                else:
                    pred = predictions[0]
                    uncertainty = uncertainties[0]
            
            # Check if we should reset state
            pred_numpy = pred.cpu().numpy()
            uncertainty_value = uncertainty.cpu().item()
            
            self.frames_since_reset += 1
            self.last_valid_distance = pred_numpy[3]
        
            # Apply smoothing if we have enough confidence
            if self.frames_since_reset > self.warm_up_frames:
                pred_numpy = self._smooth_predictions(pred_numpy)
            
            processing_time = (time.time() - processing_start) * 1000
            self.total_processing_times.append(processing_time)
            
            # Log metrics and save to CSV
            if label is not None:
                current_time = time.time()
                relative_time = current_time - self.start_time
                timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Calculate errors
                errors = self._calculate_errors(pred_numpy, label)
                
                # Prepare CSV row
                csv_row = [
                    self.frame_count, timestamp, relative_time,
                    # Predictions and ground truth
                    pred_numpy[0], pred_numpy[1], pred_numpy[2],  # pred direction
                    label[0], label[1], label[2],  # gt direction
                    pred_numpy[3], label[3],  # distance
                    pred_numpy[4], pred_numpy[5], pred_numpy[6], pred_numpy[7],  # pred rotation
                    label[4], label[5], label[6], label[7],  # gt rotation
                    pred_numpy[8], pred_numpy[9], pred_numpy[10],  # pred velocity
                    label[8], label[9], label[10],  # gt velocity
                    # Errors
                    errors['direction_error_degrees'],
                    errors['distance_error_abs'],
                    errors['distance_error_rel'],
                    errors['rotation_error_degrees'],
                    errors['velocity_error_magnitude'],
                    errors['position_error_magnitude'],
                    uncertainty_value,
                    processing_time
                ]
                
                # Save to CSV
                self._save_to_csv(csv_row)
                
                # Store for final statistics
                for key, value in errors.items():
                    if key not in self.accumulated_metrics:
                        self.accumulated_metrics[key] = []
                    self.accumulated_metrics[key].append(value)
                
                # Print progress every 50 frames
                if self.frame_count % 50 == 0:
                    print(f"\nFrame {self.frame_count} (t={relative_time:.1f}s):")
                    print(f"  Direction error: {errors['direction_error_degrees']:.2f}°")
                    print(f"  Distance error: {errors['distance_error_abs']:.3f}m")
                    print(f"  Processing: {processing_time:.1f}ms")
            
            self.frame_count += 1
            return pred_numpy.tolist()

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
        # Get predictions
        float_values = self._evaluate(img, label)

        # Send the predictions
        self.tcp_handler.send_floats(float_values, position, rotation)

    def generate_final_plots(self):
        """Generate time series plots from accumulated data"""
        if not self.accumulated_metrics:
            print("No metrics data available for plotting")
            return
        
        try:
            import pandas as pd
            
            # Read the CSV file we created
            df = pd.read_csv(self.csv_filename)
            
            # Create time series plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            error_types = [
                ('direction_error_degrees', 'Direction Error (°)'),
                ('distance_error_abs', 'Distance Error (m)'),
                ('distance_error_rel', 'Distance Error (%)'),
                ('rotation_error_degrees', 'Rotation Error (°)'),
                ('velocity_error_magnitude', 'Velocity Error (m/s)'),
                ('position_error_magnitude', 'Position Error (m)')
            ]
            
            for idx, (col, title) in enumerate(error_types):
                ax = axes[idx]
                times = df['relative_time'].values
                errors = df[col].values
                
                ax.plot(times, errors, linewidth=1, alpha=0.8)
                ax.set_title(title)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Error')
                ax.grid(True, alpha=0.3)
                
                # Add mean line
                mean_error = np.mean(errors)
                ax.axhline(y=mean_error, color='r', linestyle='--', alpha=0.7, 
                          label=f'Mean: {mean_error:.3f}')
                ax.legend()
            
            plt.tight_layout()
            plot_filename = f"results/error_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Error plots saved to: {plot_filename}")
            plt.show()
            
        except Exception as e:
            print(f"Error generating plots: {e}")

    def run(self) -> None:
        """Main execution loop"""
        try:
            print(f"Starting state-based model processor")
            print(f"CSV logging to: {self.csv_filename}")
            
            self.tcp_handler.start_receiver(self._evaluate_and_send)
            print("Model processor running. Press Ctrl+C to stop.")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            self.tcp_handler.stop()
            print("\nProcessor terminated.")

            # Print final statistics
            if self.accumulated_metrics:
                print(f"\n{'='*60}")
                print("FINAL EVALUATION RESULTS")
                print(f"{'='*60}")
                print(f"Total frames processed: {self.frame_count}")
                
                for metric, values in self.accumulated_metrics.items():
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    
                    if "degrees" in metric:
                        print(f"{metric}: {avg_value:.2f}° (±{std_value:.2f}°)")
                    elif "error" in metric:
                        print(f"{metric}: {avg_value:.3f} (±{std_value:.3f})")
                
                print(f"\nResults saved to: {self.csv_filename}")
                
                # Generate plots
                self.generate_final_plots()


if __name__ == "__main__":
    processor = StateBasedModelProcessor()
    processor.run()
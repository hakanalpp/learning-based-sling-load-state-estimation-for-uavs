import cv2
import numpy as np
import torch


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def print_test_metrics(metrics):
    print("\n" + "=" * 60)
    print("TEST-SET RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':<20}")
    print("-" * 60)

    metric_order = [                       # ← identical to final-metrics order
        ("test_loss", ".4f"),
        ("direction_angle_degrees", ".2f"),
        ("rot_loss", ".4f"),
        ("dir_loss", ".4f"),
        ("dist_loss", ".4f"),
        ("pos_loss", ".4f"),
        ("vel_loss", ".4f"),
    ]

    for name, fmt in metric_order:
        if name in metrics:
            val = metrics[name].item() if hasattr(metrics[name], "item") else metrics[name]
            if "angle" in name:
                print(f"{name:<30} {val:{fmt}}°")
            elif "vel_loss" in name:
                print(f"{name:<30} {val:{fmt}} (velocity)")
            elif "pos_loss" in name:
                print(f"{name:<30} {val:{fmt}} (position)")
            else:
                print(f"{name:<30} {val:{fmt}}")

    print("=" * 60)


def check_for_nan_values(metrics, pred, gt_tensor, label):
    """Check for NaN values and print debugging information"""
    has_nan = False

    # Check metrics for NaN
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"⚠️  WARNING: NaN found in {key}")
                has_nan = True
        elif isinstance(value, float) and np.isnan(value):
            print(f"⚠️  WARNING: NaN found in {key}")
            has_nan = True

    # Check predictions for NaN
    if torch.isnan(pred).any():
        print(f"⚠️  WARNING: NaN found in predictions")
        print(f"   Prediction values: {pred.squeeze().cpu().numpy()}")
        has_nan = True

    # Check ground truth for NaN/invalid values
    if torch.isnan(gt_tensor).any():
        print(f"⚠️  WARNING: NaN found in ground truth")
        print(f"   GT label values: {label}")
        has_nan = True

    # Check for invalid values in ground truth
    direction = gt_tensor[0, :3]
    distance = gt_tensor[0, 3]
    quaternion = gt_tensor[0, 4:8]

    if torch.norm(direction) < 1e-6:
        print(f"⚠️  WARNING: Direction vector is nearly zero: {direction.cpu().numpy()}")
        has_nan = True

    if distance <= 0:
        print(f"⚠️  WARNING: Distance is non-positive: {distance.item()}")
        has_nan = True

    if torch.norm(quaternion) < 1e-6:
        print(f"⚠️  WARNING: Quaternion is nearly zero: {quaternion.cpu().numpy()}")
        has_nan = True

    # If we found issues, print additional debug info
    if has_nan:
        print(f"   Input label: {label}")
        print(f"   Direction norm: {torch.norm(direction).item():.6f}")
        print(f"   Quaternion norm: {torch.norm(quaternion).item():.6f}")
        print(f"   Distance value: {distance.item():.6f}")


def print_final_metrics(frame_count, accumulated_metrics):
    """Print averaged metrics in a nice format"""

    avg_metrics = calculate_average_metrics(accumulated_metrics)

    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION RESULTS (over {frame_count} frames)")
    print("=" * 60)
    print(f"{'Metric':<30} {'Average Value':<20}")
    print("-" * 60)

    # Define the order and format for essential metrics
    metric_order = [
        ("test_loss", ".4f"),
        ("direction_angle_degrees", ".2f"),
        ("rot_loss", ".4f"),
        ("dir_loss", ".4f"),
        ("dist_loss", ".4f"),
        ("pos_loss", ".4f"),
        ("vel_loss", ".4f"),
    ]

    for metric_name, fmt in metric_order:
        if metric_name in avg_metrics:
            value = avg_metrics[metric_name]
            if "angle" in metric_name:
                print(f"{metric_name:<30} {value:{fmt}}°")
            elif "vel_loss" in metric_name:
                print(f"{metric_name:<30} {value:{fmt}} (velocity)")
            elif "pos_loss" in metric_name:
                print(f"{metric_name:<30} {value:{fmt}} (position)")
            else:
                print(f"{metric_name:<30} {value:{fmt}}")

    print("=" * 60)
    print(f"Total frames processed: {frame_count}")


def calculate_average_metrics(accumulated_metrics):
    avg_metrics = {}
    for key, values in accumulated_metrics.items():
        avg_metrics[key] = sum(values) / len(values)

    return avg_metrics


def accumulate_metrics(accumulated_metrics, metrics):
    """Accumulate metrics for averaging later"""
    for key, value in metrics.items():
        if key not in accumulated_metrics:
            accumulated_metrics[key] = []

        # Convert tensor to scalar if needed
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:  # Single value tensor
                accumulated_metrics[key].append(value.cpu().item())
            else:  # Multi-value tensor (like direction_angle_degrees)
                accumulated_metrics[key].append(value.mean().cpu().item())
        else:
            accumulated_metrics[key].append(value)

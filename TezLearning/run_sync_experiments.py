import itertools
import json
import math
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from dataloader import StateBasedDroneDataset

pl.seed_everything(42)


class ImprovedDroneModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Choose backbone based on config
        if config["backbone"] == "resnet34":
            backbone = models.resnet34(weights="IMAGENET1K_V1")
            feature_dim = 512
        elif config["backbone"] == "resnet50":
            backbone = models.resnet50(weights="IMAGENET1K_V1")
            feature_dim = 2048
        elif config["backbone"] == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
            feature_dim = 1280

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Adaptive pooling for different backbones
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Build head with configurable depth
        layers = []
        in_dim = feature_dim

        for hidden_dim in config["hidden_dims"]:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config["dropout_rate"]),
                ]
            )
            in_dim = hidden_dim

        # Final layer - now outputs 11 values: 3 dir + 1 dist + 4 quat
        layers.append(nn.Linear(in_dim, 11))
        self.head = nn.Sequential(*layers)

        # Loss weights
        self.dir_weight = config["dir_weight"]
        self.dist_weight = config["dist_weight"]
        self.rot_weight = config["rot_weight"]

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features).squeeze(-1).squeeze(-1)
        out = self.head(features)

        direction = F.normalize(out[:, :3], dim=-1)
        distance = F.softplus(out[:, 3:4])
        quaternion = F.normalize(out[:, 4:8], dim=-1)

        return torch.cat([direction, distance, quaternion, out[:, 8:11]], dim=-1)

    def configure_optimizers(self):
        if self.config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["learning_rate"],
                momentum=0.9,
                weight_decay=self.config["weight_decay"],
            )

        if self.config["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["scheduler_t_max"]
            )
        elif self.config["scheduler"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        else:  # reduce_on_plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )

        if self.config["scheduler"] == "reduce_on_plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            return [optimizer], [scheduler]

    def direction_loss(self, pred_dir, target_dir):
        return 1 - F.cosine_similarity(pred_dir, target_dir, dim=-1).mean()

    def distance_loss(self, pred_dist, target_dist):
        if self.config.get("distance_loss_type", "mse") == "mse":
            return F.mse_loss(pred_dist, target_dist)
        else:  # huber
            return F.huber_loss(pred_dist, target_dist)

    def quaternion_geodesic_loss(self, pred_quat, target_quat):
        dot = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
        dot = torch.clamp(dot, 0.0, 1.0)
        return (2 * torch.acos(dot)).mean()

    def compute_loss(self, pred, gt):
        pred_dir = pred[:, 0:3]
        pred_dist = pred[:, 3:4]
        pred_quat = pred[:, 4:8]

        gt_dir = gt[:, 0:3]
        gt_dist = gt[:, 3:4]
        gt_quat = gt[:, 4:8]

        dir_loss = self.direction_loss(pred_dir, gt_dir)
        dist_loss = self.distance_loss(pred_dist, gt_dist)
        rot_loss = self.quaternion_geodesic_loss(pred_quat, gt_quat)

        total_loss = (
            self.dir_weight * dir_loss
            + self.dist_weight * dist_loss
            + self.rot_weight * rot_loss
        )

        return total_loss, {
            "dir_loss": dir_loss,
            "dist_loss": dist_loss,
            "rot_loss": rot_loss,
        }

    def training_step(self, batch, _):
        image = batch["image"]
        gt = batch["label"]
        pred = self(image)
        loss, components = self.compute_loss(pred, gt)

        self.log("train_loss", loss)
        self.log_dict({f"train_{k}": v for k, v in components.items()})
        return loss

    def validation_step(self, batch, _):
        image = batch["image"]
        target = batch["label"]

        with torch.no_grad():
            pred = self(image)
            loss, components = self.compute_loss(pred, target)

        self.log("val_loss", loss, prog_bar=True)
        self.log_dict({f"val_{k}": v for k, v in components.items()})
        return loss

    def test_step(self, batch, _):
        image = batch["image"]
        gt = batch["label"]

        with torch.no_grad():
            pred = self(image)
            loss, loss_components = self.compute_loss(pred, gt)

        pred_dir = pred[:, 0:3]
        pred_dist = pred[:, 3:4]
        pred_quat = pred[:, 4:8]

        gt_dir = gt[:, 0:3]
        gt_dist = gt[:, 3:4]
        gt_quat = gt[:, 4:8]

        # Calculate detailed metrics
        cos_angle = torch.clamp(F.cosine_similarity(pred_dir, gt_dir, dim=-1), -1, 1)
        direction_angle_error_deg = (
            torch.acos(torch.abs(cos_angle)).mean() * 180.0 / math.pi
        )

        distance_error_m = torch.abs(pred_dist - gt_dist).mean()

        dot_product = torch.abs(torch.sum(pred_quat * gt_quat, dim=-1))
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        rotation_angle_error_deg = (
            (2 * torch.acos(dot_product)).mean() * 180.0 / math.pi
        )

        metrics = {
            "test_loss": loss,
            **{f"test_{k}": v for k, v in loss_components.items()},
            "test_direction_angle_error_deg": direction_angle_error_deg,
            "test_distance_error_m": distance_error_m,
            "test_rotation_angle_error_deg": rotation_angle_error_deg,
        }

        self.log_dict(metrics, prog_bar=True)
        return metrics


def create_datasets(augment_train=False):
    # Use the original dataset class since we don't have access to modify it
    dataset = StateBasedDroneDataset(
        images_folder="/home/alp/noetic_ws/TezLearning/data/images",
        csv_path="/home/alp/noetic_ws/TezLearning/data/images/cargo_data.csv",
    )

    total_sequences = len(dataset)
    n_train = int(0.7 * total_sequences)
    n_val = int(0.15 * total_sequences)

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, total_sequences))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def check_existing_experiments():
    """Check which experiments are already completed"""
    completed = set()

    # Check for existing result files
    for file in os.listdir("."):
        if file.startswith("hp_search_results_") and file.endswith(".json"):
            try:
                with open(file, "r") as f:
                    results = json.load(f)
                    for result in results:
                        if "best_val_loss" in result and result[
                            "best_val_loss"
                        ] != float("inf"):
                            completed.add(result["experiment_id"])
            except:
                continue

    # Check for checkpoint directories
    if os.path.exists("hp_search_checkpoints"):
        for exp_dir in os.listdir("hp_search_checkpoints"):
            if exp_dir.startswith("exp_"):
                exp_id = int(exp_dir.split("_")[1])
                # Check if there are any checkpoint files
                exp_path = os.path.join("hp_search_checkpoints", exp_dir)
                if os.path.exists(exp_path) and os.listdir(exp_path):
                    completed.add(exp_id)

    return completed


def run_experiment(config, experiment_id, run_test=True):
    print(f"\n{'=' * 50}")
    print(f"Starting Experiment {experiment_id}")
    print(f"Config: {config}")
    print(f"{'=' * 50}")

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        config.get("augment", False)
    )

    # Create data loaders with higher num_workers for speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=config.get("num_workers", 16),
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=config.get("num_workers", 16),
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=config.get("num_workers", 16),
        persistent_workers=True,
    )

    # Create model
    model = ImprovedDroneModel(config)

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"exp_{experiment_id}-{{epoch:02d}}-{{val_loss:.4f}}",
        dirpath=f"./hp_search_checkpoints/exp_{experiment_id}",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config.get("patience", 8), mode="min"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 60),
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        precision=config.get("precision", "16-mixed"),
        enable_progress_bar=True,  # Reduce console spam
        enable_model_summary=True,
        log_every_n_steps=20,  # Reduce logging frequency
    )

    try:
        # Train
        trainer.fit(model, train_loader, val_loader)

        # Get best validation loss
        best_val_loss = checkpoint_callback.best_model_score.item()

        # Run test if requested
        test_results = None
        if run_test:
            print(f"Running test for experiment {experiment_id}...")
            test_results = trainer.test(
                model, test_loader, ckpt_path=checkpoint_callback.best_model_path
            )
            test_results = test_results[0] if test_results else None

        # Save results
        result = {
            "experiment_id": experiment_id,
            "config": config,
            "best_val_loss": best_val_loss,
            "epochs_trained": trainer.current_epoch,
            "checkpoint_path": checkpoint_callback.best_model_path,
            "test_results": test_results,
        }

        return result

    except Exception as e:
        print(f"Experiment {experiment_id} failed: {str(e)}")
        return {
            "experiment_id": experiment_id,
            "config": config,
            "best_val_loss": float("inf"),
            "error": str(e),
        }


def main():
    # Check which experiments are already done
    completed_experiments = check_existing_experiments()
    print(
        f"Found {len(completed_experiments)} completed experiments: {sorted(completed_experiments)}"
    )

    # Create results directory
    os.makedirs("hp_search_checkpoints", exist_ok=True)

    # Based on your results, let's focus on the best performing configurations
    # Experiment 1 was best (0.225 val loss) - ResNet50 + 1e-4 LR + ReduceLROnPlateau
    promising_configs = [
        # Recreate the EXACT previous winning config (Exp 5 from before)
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [256, 128], 'dropout_rate': 0.2, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-4,
            'gradient_clip_val': 0.5, 'dir_weight': 2.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Winner + EXTREME direction focus (break the 1° barrier!)
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [256, 128], 'dropout_rate': 0.2, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-4,
            'gradient_clip_val': 0.5, 'dir_weight': 5.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Winner + minimal regularization (like current Exp 16)
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [256, 128], 'dropout_rate': 0.05, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-5,
            'gradient_clip_val': 0.3, 'dir_weight': 2.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Winner + wider network (like current Exp 17)
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [512, 256], 'dropout_rate': 0.2, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-4,
            'gradient_clip_val': 0.5, 'dir_weight': 2.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Winner + larger batch for better GPU utilization
        {
            'learning_rate': 1e-4, 'batch_size': 96, 'backbone': 'resnet50',
            'hidden_dims': [256, 128], 'dropout_rate': 0.2, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-4,
            'gradient_clip_val': 0.5, 'dir_weight': 2.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Winner + cosine scheduler
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [256, 128], 'dropout_rate': 0.2, 'optimizer': 'adamw',
            'scheduler': 'cosine', 'scheduler_t_max': 60, 'weight_decay': 1e-4,
            'gradient_clip_val': 0.5, 'dir_weight': 2.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Hybrid: minimal reg + extreme direction focus
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [256, 128], 'dropout_rate': 0.05, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-5,
            'gradient_clip_val': 0.3, 'dir_weight': 4.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Hybrid: wider network + extreme direction focus
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [512, 256], 'dropout_rate': 0.2, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-4,
            'gradient_clip_val': 0.5, 'dir_weight': 4.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 12, 'max_epochs': 60, 'precision': "16-mixed", 'num_workers': 16
        },
        # Ultimate hybrid: best of everything
        {
            'learning_rate': 1e-4, 'batch_size': 64, 'backbone': 'resnet50',
            'hidden_dims': [512, 256], 'dropout_rate': 0.05, 'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau', 'scheduler_t_max': 60, 'weight_decay': 1e-5,
            'gradient_clip_val': 0.3, 'dir_weight': 3.0, 'dist_weight': 1.0,
            'rot_weight': 1.0, 'distance_loss_type': 'mse',
            'augment': False, 'patience': 15, 'max_epochs': 70, 'precision': "16-mixed", 'num_workers': 16
        }
    ]
    # Run experiments, skipping completed ones
    results = []

    # Load existing results if any
    for file in os.listdir("."):
        if file.startswith("hp_search_results_") and file.endswith(".json"):
            try:
                with open(file, "r") as f:
                    existing_results = json.load(f)
                    results.extend(existing_results)
                    break
            except:
                continue

    for i, config in enumerate(promising_configs):
        if i in completed_experiments:
            print(f"Skipping experiment {i} (already completed)")
            continue

        result = run_experiment(config, i, run_test=True)
        results.append(result)

        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"hp_search_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(
            f"Experiment {i} completed. Best val loss: {result.get('best_val_loss', 'Failed')}"
        )
        if "test_results" in result and result["test_results"]:
            test_res = result["test_results"]
            print(f"  Test metrics:")
            print(
                f"    Direction error: {test_res.get('test_direction_angle_error_deg', 'N/A'):.2f}°"
            )
            print(
                f"    Distance error: {test_res.get('test_distance_error_m', 'N/A'):.4f}m"
            )
            print(
                f"    Rotation error: {test_res.get('test_rotation_angle_error_deg', 'N/A'):.2f}°"
            )

    # Sort results by validation loss
    successful_results = [r for r in results if "error" not in r]
    successful_results.sort(key=lambda x: x["best_val_loss"])

    print(f"\n{'=' * 60}")
    print("FAST HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(successful_results)}")

    if successful_results:
        print(f"\nTop 3 configurations:")
        for i, result in enumerate(successful_results[:3]):
            print(f"\n{i + 1}. Val Loss: {result['best_val_loss']:.6f}")
            print(f"   Epochs: {result['epochs_trained']}")
            if "test_results" in result and result["test_results"]:
                test_res = result["test_results"]
                print(
                    f"   Test - Dir: {test_res.get('test_direction_angle_error_deg', 'N/A'):.2f}°, "
                    f"Dist: {test_res.get('test_distance_error_m', 'N/A'):.4f}m, "
                    f"Rot: {test_res.get('test_rotation_angle_error_deg', 'N/A'):.2f}°"
                )
            print(f"   Config: {result['config']}")

    # Save final results
    final_results_file = (
        f"final_fast_hp_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(final_results_file, "w") as f:
        json.dump(
            {"all_results": results, "best_configs": successful_results[:5]},
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to: {final_results_file}")


if __name__ == "__main__":
    main()

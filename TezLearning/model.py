import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DroneModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Adaptive pooling for different backbones
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Experiment 7's winning head: [512, 256] with dropout 0.2
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8),  # 3 direction + 1 distance + 4 quaternion + 3 dummy
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features).squeeze(-1).squeeze(-1)
        out = self.head(features)

        direction = F.normalize(out[:, :3], dim=-1)
        distance = F.softplus(out[:, 3:4])
        quaternion = F.normalize(out[:, 4:8], dim=-1)

        return torch.cat(
            [direction, distance, quaternion], dim=-1
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def direction_loss(self, pred_dir, target_dir):
        """Cosine similarity loss for direction"""
        return 1 - F.cosine_similarity(pred_dir, target_dir, dim=-1).mean()

    def distance_loss(self, pred_dist, target_dist):
        """Mean squared error loss for distance"""
        return F.mse_loss(pred_dist, target_dist)

    def quaternion_geodesic_loss(self, pred_quat, target_quat):
        """Geodesic distance loss for quaternions"""
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

        total_loss = 4.0 * dir_loss + dist_loss + rot_loss

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

        cos_angle = torch.clamp(F.cosine_similarity(pred_dir, gt_dir, dim=-1), -1, 1)
        direction_angle_error_deg = (
            torch.acos(torch.abs(cos_angle)).mean() * 180.0 / np.pi
        )

        distance_error_m = torch.abs(pred_dist - gt_dist).mean()

        dot_product = torch.abs(torch.sum(pred_quat * gt_quat, dim=-1))
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        rotation_angle_error_deg = (2 * torch.acos(dot_product)).mean() * 180.0 / np.pi

        self.log_dict(
            {
                "test_loss": loss,
                **{f"test_{k}": v for k, v in loss_components.items()},
                "test_direction_angle_error_deg": direction_angle_error_deg,
                "test_distance_error_m": distance_error_m,
                "test_rotation_angle_error_deg": rotation_angle_error_deg,
            },
            prog_bar=True,
        )

        return {
            "loss": loss,
            **{f"test_{k}": v for k, v in loss_components.items()},
            "direction_angle_error_deg": direction_angle_error_deg,
            "distance_error_m": distance_error_m,
            "rotation_angle_error_deg": rotation_angle_error_deg,
        }

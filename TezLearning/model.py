import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CargoPoseModel(pl.LightningModule):
    def __init__(self, λ_rot=0.8, λ_dir=0.5, λ_dist=0.1, λ_pos=1.5):
        super().__init__()
        backbone = models.resnet34(weights="IMAGENET1K_V1")
        self.head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 8),  # 8 outputs: 3 direction, 1 distance, 4 quaternion
        )
        backbone.fc = self.head
        self.model = backbone
        self.λ_rot = λ_rot
        self.λ_dir = λ_dir
        self.λ_dist = λ_dist
        self.λ_pos = λ_pos  # Higher weight for position error to prioritize it

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def _quad_loss(self, pred_quat, gt_quat):
        q1 = F.normalize(pred_quat, dim=-1)
        q2 = F.normalize(gt_quat, dim=-1)
        dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))

        quat_loss = 1.0 - dot_product
        return quat_loss.mean()

    def _geodesic_loss(self, pred_quat, gt_quat):
        q1 = F.normalize(pred_quat, dim=-1)
        q2 = F.normalize(gt_quat, dim=-1)
        dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        loss = torch.acos(dot)
        return loss.mean()

    def _direction_loss(self, pred_v, gt_v):
        v1 = F.normalize(pred_v, dim=-1)
        v2 = F.normalize(gt_v, dim=-1)
        cos = torch.sum(v1 * v2, dim=-1).clamp(-1.0, 1.0)  # cosθ
        return (1.0 - cos).mean()  # 0 when aligned

    def _distance_loss(self, pred_d, gt_d):
        return F.l1_loss(pred_d.squeeze(-1), gt_d.squeeze(-1))  # metres

    def position_error(self, pred, gt):
        """
        Calculate the error between predicted and ground truth positions in 3D space.
        Applies additional penalty for large errors to handle edge cases better.

        Args:
            pred: Model prediction with shape [..., 8] (3 direction, 1 distance, 4 quaternion)
            gt: Ground truth with shape [..., 8] (3 direction, 1 distance, 4 quaternion)

        Returns:
            Weighted position error with higher penalties for large errors
        """
        pred_direction = F.normalize(pred[:, 0:3], dim=-1)
        pred_dist = F.softplus(pred[:, 3:4])

        gt_direction = F.normalize(gt[:, 0:3], dim=-1)
        gt_dist = gt[:, 3:4]

        pred_position = pred_direction * pred_dist
        gt_position = gt_direction * gt_dist

        position_errors = torch.norm(pred_position - gt_position, dim=1)

        threshold = 0.5  # threshold in meters
        edge_case_errors = F.relu(position_errors - threshold) ** 2

        weighted_errors = position_errors + 0.5 * edge_case_errors

        return weighted_errors.mean()

    def loss(self, pred, gt):
        pred_direction, pred_dist, pred_quat = (
            pred[:, 0:3],
            F.softplus(pred[:, 3:4]),
            pred[:, 4:8],
        )
        gt_direction, gt_dist, gt_quat = gt[:, 0:3], gt[:, 3:4], gt[:, 4:8]

        L_rot = self._geodesic_loss(pred_quat, gt_quat)
        L_dir = self._direction_loss(pred_direction, gt_direction)
        L_dist = self._distance_loss(pred_dist, gt_dist)
        L_pos = self.position_error(pred, gt)

        loss = (
            self.λ_rot * L_rot
            + self.λ_dir * L_dir
            + self.λ_dist * L_dist
            + self.λ_pos * L_pos
        )

        return loss, {
            "rot_loss": L_rot,
            "dir_loss": L_dir,
            "dist_loss": L_dist,
            "pos_loss": L_pos,
        }

    def quat_angle_error_deg(self, pred_quat, gt_quat):
        q1 = F.normalize(pred_quat, dim=-1)
        q2 = F.normalize(gt_quat, dim=-1)
        dot = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
        angle_rad = 2 * torch.acos(torch.abs(dot))
        return torch.rad2deg(angle_rad)

    def training_step(self, batch, _):
        images = batch["image"]
        gt = batch["label"]
        preds = self(images)
        loss = self.loss(preds, gt)
        self.log("train_loss", loss[0])
        self.log_dict({f"train_{k}": v for k, v in loss[1].items()}, on_epoch=True)
        return loss[0]

    def validation_step(self, batch, _):
        images = batch["image"]
        gt = batch["label"]
        preds = self(images)
        val_loss = self.loss(preds, gt)
        pos_error = self.position_error(preds, gt)

        # Log losses and metrics
        self.log("val_loss", val_loss[0], prog_bar=True)
        self.log("val_pos_error_m", pos_error, prog_bar=True)
        self.log_dict(
            {f"val_{k}": v for k, v in val_loss[1].items()},
            on_epoch=True,
            prog_bar=True,
        )

        # Calculate extreme position errors (>1m) to monitor edge cases
        extreme_errors = (
            (
                torch.norm(
                    F.normalize(preds[:, 0:3], dim=-1) * F.softplus(preds[:, 3:4])
                    - F.normalize(gt[:, 0:3], dim=-1) * gt[:, 3:4],
                    dim=1,
                )
                > 0.2
            )
            .float()
            .mean()
        )

        self.log("val_extreme_errors", extreme_errors, prog_bar=True)

        return val_loss[0]

    def test_step(self, batch, _):
        x = batch["image"]
        gt = batch["label"]
        pred = self(x)

        loss = self.loss(pred, gt)
        angle_error = self.quat_angle_error_deg(pred[:, 4:8], gt[:, 4:8])
        pos_error = self.position_error(pred, gt)

        # Calculate position errors per sample for analysis
        pred_position = F.normalize(pred[:, 0:3], dim=-1) * F.softplus(pred[:, 3:4])
        gt_position = F.normalize(gt[:, 0:3], dim=-1) * gt[:, 3:4]
        position_errors = torch.norm(pred_position - gt_position, dim=1)

        # Log standard metrics
        self.log("test_loss", loss[0])
        self.log("test_angle_deg", angle_error.mean())
        self.log("test_pos_error_m", pos_error)
        self.log_dict({f"test_{k}": v for k, v in loss[1].items()}, on_epoch=True)

        # Log detailed error statistics
        self.log("test_pos_error_max", position_errors.max())
        self.log("test_pos_error_median", position_errors.median())
        self.log("test_pos_error_90percentile", torch.quantile(position_errors, 0.9))

        # Count errors above thresholds
        self.log("test_errors_above_0.5m", (position_errors > 0.5).float().mean())
        self.log("test_errors_above_1.0m", (position_errors > 1.0).float().mean())

        return {
            "loss": loss[0],
            "angle_error": angle_error,
            "pos_error": pos_error,
            "position_errors": position_errors,  # Return individual errors for post-processing
        }

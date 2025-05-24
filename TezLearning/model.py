import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Model(pl.LightningModule):
    def __init__(
        self, λ_rot=0.8, λ_dir=0.5, λ_dist=0.1, λ_pos=1.5, λ_vel=0.3, sequence_length=5
    ):
        super().__init__()

        # Smaller backbone to reduce overfitting
        backbone = models.resnet34(weights="IMAGENET1K_V1")
        cnn_features = backbone.fc.in_features  # 512 features
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Add dropout to feature extractor
        self.feature_dropout = nn.Dropout(0.2)

        # Smaller RNN with dropout
        self.rnn = nn.RNN(
            input_size=cnn_features,
            hidden_size=128,  # Reduced from 256
            num_layers=1,
            batch_first=True,
            dropout=0.4,
        )

        # Regularized head
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 11),
        )

        self.λ_rot = λ_rot
        self.λ_dir = λ_dir
        self.λ_dist = λ_dist
        self.λ_pos = λ_pos
        self.λ_vel = λ_vel
        self.sequence_length = sequence_length

    def forward(self, x):
        """
        Forward pass for image sequences
        x shape: [batch_size, sequence_length, channels, height, width]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Reshape to process all frames at once
        x_reshaped = x.view(batch_size * seq_len, *x.shape[2:])

        # Extract features for all frames at once
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size * seq_len, -1)
        features = self.feature_dropout(features)  # Add dropout after features

        # Reshape back to sequences
        sequence_features = features.view(batch_size, seq_len, -1)

        # Process through RNN
        rnn_out, _ = self.rnn(sequence_features)
        final_features = rnn_out[:, -1]  # Use last timestep

        # Generate predictions
        outputs = self.head(final_features)
        return outputs

    def configure_optimizers(self):
        """Configure optimizer with more aggressive scheduling"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=5e-4, weight_decay=1e-4
        )  # Add weight decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.3,  # More aggressive reduction
            patience=5,  # Shorter patience
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def _rotation_loss(self, pred_quat, gt_quat):
        """Geodesic loss for quaternion rotation"""
        q1 = F.normalize(pred_quat, dim=-1)
        q2 = F.normalize(gt_quat, dim=-1)
        dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(1e-7, 1.0 - 1e-7)
        return torch.acos(dot).mean()

    def _direction_loss(self, pred_dir, gt_dir):
        """Angular loss for direction vectors"""
        v1 = F.normalize(pred_dir, dim=-1)
        v2 = F.normalize(gt_dir, dim=-1)
        cos_sim = torch.sum(v1 * v2, dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cos_sim).mean()

    def _distance_loss(self, pred_dist, gt_dist):
        """L1 loss for distance prediction"""
        return F.l1_loss(pred_dist.squeeze(-1), gt_dist.squeeze(-1))

    def _velocity_loss(self, pred_vel, gt_vel):
        """L1 loss for velocity prediction"""
        return F.l1_loss(pred_vel, gt_vel)

    def _position_loss(self, pred, gt):
        """Combined position loss with edge case handling"""
        pred_dir = F.normalize(pred[:, 0:3], dim=-1)
        pred_dist = pred[:, 3:4]
        gt_dir = F.normalize(gt[:, 0:3], dim=-1)
        gt_dist = gt[:, 3:4]

        pred_pos = pred_dir * pred_dist
        gt_pos = gt_dir * gt_dist

        position_errors = torch.norm(pred_pos - gt_pos, dim=1)

        threshold = 0.5
        edge_penalties = F.relu(position_errors - threshold) ** 2
        weighted_errors = position_errors + 0.5 * edge_penalties

        return weighted_errors.mean()

    def _direction_angle_degrees(self, pred_dir, gt_dir):
        """Calculate angle error between direction vectors in degrees"""
        v1 = F.normalize(pred_dir, dim=-1)
        v2 = F.normalize(gt_dir, dim=-1)
        dot = torch.sum(v1 * v2, dim=-1).clamp(-1.0, 1.0)
        angle_rad = torch.acos(dot)
        return torch.rad2deg(angle_rad)

    def loss(self, pred, gt):
        """Compute weighted loss combination"""
        pred_dir, pred_dist, pred_quat, pred_vel = (
            pred[:, 0:3],
            pred[:, 3:4],
            pred[:, 4:8],
            pred[:, 8:11],
        )
        gt_dir, gt_dist, gt_quat, gt_vel = (
            gt[:, 0:3],
            gt[:, 3:4],
            gt[:, 4:8],
            gt[:, 8:11],
        )

        L_rot = self._rotation_loss(pred_quat, gt_quat)
        L_dir = self._direction_loss(pred_dir, gt_dir)
        L_dist = self._distance_loss(pred_dist, gt_dist)
        L_pos = self._position_loss(pred, gt)
        L_vel = self._velocity_loss(pred_vel, gt_vel)

        total_loss = (
            self.λ_rot * L_rot
            + self.λ_dir * L_dir
            + self.λ_dist * L_dist
            + self.λ_pos * L_pos
            + self.λ_vel * L_vel
        )

        loss_components = {
            "rot_loss": L_rot,
            "dir_loss": L_dir,
            "dist_loss": L_dist,
            "pos_loss": L_pos,
            "vel_loss": L_vel,
        }

        return total_loss, loss_components

    def training_step(self, batch, _):
        images = batch["image_sequence"]
        labels = batch["label"]

        predictions = self(images)
        loss, loss_components = self.loss(predictions, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(
            {f"train_{k}": v for k, v in loss_components.items()}, on_epoch=True
        )

        return loss

    def validation_step(self, batch, _):
        images = batch["image_sequence"]
        labels = batch["label"]

        predictions = self(images)
        loss, loss_components = self.loss(predictions, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(
            {f"val_{k}": v for k, v in loss_components.items()},
        )

        return loss

    def test_step(self, batch, _):
        images = batch["image_sequence"]
        labels = batch["label"]

        predictions = self(images)
        loss, loss_components = self.loss(predictions, labels)

        direction_angle_degrees = self._direction_angle_degrees(
            predictions[:, 0:3], labels[:, 0:3]
        )

        self.log("test_loss", loss)
        self.log_dict(
            {
                "direction_angle_degrees": direction_angle_degrees.mean(),
                **loss_components,
            }
        )

        return {
            "test_loss": loss,
            "direction_angle_degrees": direction_angle_degrees.mean(),
            **loss_components,
        }

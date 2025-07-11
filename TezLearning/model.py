import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class StateBasedModel(pl.LightningModule):
    def __init__(
        self, 
        λ_rot=0.8, 
        λ_dir=1.0, 
        λ_dist=0.1, 
        λ_pos=1.5, 
        λ_vel=0.3,
        λ_state_reg=0.1,
        λ_temporal=0.2,
        hidden_dim=256,
        state_dim=128,
        dropout_rate=0.3
    ):
        super().__init__()
        
        self.λ_rot = λ_rot
        self.λ_dir = λ_dir
        self.λ_dist = λ_dist
        self.λ_pos = λ_pos
        self.λ_vel = λ_vel
        self.λ_state_reg = λ_state_reg
        self.λ_temporal = λ_temporal
        
        # Architecture parameters
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Feature extractor (lighter backbone for efficiency)
        backbone = models.resnet34(weights="IMAGENET1K_V1")
        self.feature_dim = backbone.fc.in_features  # 512
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # Freeze early layers to prevent overfitting
        for param in list(self.feature_extractor.parameters())[:30]:
            param.requires_grad = False
            
        # Image feature processing
        self.image_processor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # State update network (GRU-based)
        self.state_updater = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=state_dim
        )
        
        # State-to-output decoders
        self.direction_decoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Direction vector
        )
        
        self.distance_decoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),  # Distance
            nn.Softplus()  # Ensure positive distance
        )
        
        self.rotation_decoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4)  # Quaternion
        )
        
        self.velocity_decoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)  # Velocity vector
        )
        
        # Uncertainty estimation heads
        self.direction_uncertainty = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # Initialize biases for stable predictions
        self._initialize_biases()
        
    def _initialize_biases(self):
        """Initialize biases for stable initial predictions"""
        # Distance decoder should start with reasonable distance
        with torch.no_grad():
            self.distance_decoder[-2].bias.fill_(1.0)
            
    def forward(self, x, initial_state=None, return_all_states=False):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
            
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if initial_state is None:
            state = torch.zeros(batch_size, self.state_dim, device=x.device)
        else:
            state = initial_state
            
        outputs = []
        states = []
        uncertainties = []
        
        for t in range(seq_len):
            frame = x[:, t]
            features = self.feature_extractor(frame)
            features = features.view(batch_size, -1)
            
            processed_features = self.image_processor(features)
            
            state = self.state_updater(processed_features, state)
            states.append(state)
            
            direction = self.direction_decoder(state)
            direction = F.normalize(direction, dim=-1)
            
            distance = self.distance_decoder(state)
            rotation = self.rotation_decoder(state)
            rotation = F.normalize(rotation, dim=-1)
            
            velocity = self.velocity_decoder(state)
            
            dir_uncertainty = self.direction_uncertainty(state)
            uncertainties.append(dir_uncertainty)
            
            output = torch.cat([direction, distance, rotation, velocity], dim=-1)
            outputs.append(output)
            
        outputs = torch.stack(outputs, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)
        
        if seq_len == 1:
            outputs = outputs.squeeze(1)
            uncertainties = uncertainties.squeeze(1)
            
        if return_all_states:
            return outputs, states, uncertainties
        else:
            return outputs
            
    def _robust_direction_loss(self, pred_dir, gt_dir, uncertainties=None):
        pred_dir_norm = F.normalize(pred_dir, dim=-1)
        gt_dir_norm = F.normalize(gt_dir, dim=-1)
        
        cos_sim = torch.sum(pred_dir_norm * gt_dir_norm, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        angle_error = torch.acos(cos_sim)
        
        threshold = 0.3
        small_error_mask = angle_error < threshold
        
        loss = torch.where(
            small_error_mask,
            angle_error ** 2,
            threshold * torch.sqrt(angle_error)
        )
        
        if uncertainties is not None:
            weights = 1.0 / (1.0 + uncertainties.squeeze(-1))
            loss = loss * weights
            
        return loss.mean()
        
    def _temporal_consistency_loss(self, predictions):
        if predictions.shape[1] < 2:
            return 0.0
        
        diffs = predictions[:, 1:] - predictions[:, :-1]
        
        dir_diffs = torch.norm(diffs[:, :, :3], dim=-1)
        dist_diffs = torch.abs(diffs[:, :, 3])
        
        dir_penalty = torch.mean(dir_diffs[dir_diffs > 0.1])
        dist_penalty = torch.mean(dist_diffs[dist_diffs > 0.5])
        
        return dir_penalty + 0.5 * dist_penalty if not torch.isnan(dir_penalty) else 0.0
        
    def loss(self, pred, gt, states=None, uncertainties=None, full_sequence_pred=None):
        if len(pred.shape) == 3:
            pred = pred[:, -1]
            if uncertainties is not None:
                uncertainties = uncertainties[:, -1]
                
        pred_dir = pred[:, 0:3]
        pred_dist = pred[:, 3:4]
        pred_quat = pred[:, 4:8]
        pred_vel = pred[:, 8:11]
        
        gt_dir = gt[:, 0:3]
        gt_dist = gt[:, 3:4]
        gt_quat = gt[:, 4:8]
        gt_vel = gt[:, 8:11]
        
        L_dir = self._robust_direction_loss(pred_dir, gt_dir, uncertainties)
        L_dist = F.smooth_l1_loss(pred_dist, gt_dist)
        L_rot = self._rotation_loss(pred_quat, gt_quat)
        L_vel = F.smooth_l1_loss(pred_vel, gt_vel)
        
        pred_pos = pred_dir * pred_dist
        gt_pos = gt_dir * gt_dist
        L_pos = F.smooth_l1_loss(pred_pos, gt_pos)
        
        L_state = 0.0
        if states is not None:
            state_changes = []
            for i in range(1, len(states)):
                change = torch.norm(states[i] - states[i-1], dim=-1)
                state_changes.append(change)
            if state_changes:
                L_state = torch.mean(torch.stack(state_changes))
        
        L_temporal = 0.0
        if full_sequence_pred is not None:
            L_temporal = self._temporal_consistency_loss(full_sequence_pred)
            
        total_loss = (
            self.λ_dir * L_dir +
            self.λ_dist * L_dist +
            self.λ_rot * L_rot +
            self.λ_vel * L_vel +
            self.λ_pos * L_pos +
            self.λ_state_reg * L_state +
            self.λ_temporal * L_temporal
        )
        
        loss_components = {
            "dir_loss": L_dir,
            "dist_loss": L_dist,
            "rot_loss": L_rot,
            "vel_loss": L_vel,
            "pos_loss": L_pos,
            "state_loss": L_state,
            "temporal_loss": L_temporal
        }
        
        return total_loss, loss_components
        
    def _rotation_loss(self, pred_quat, gt_quat):
        """Geodesic rotation loss"""
        q1 = F.normalize(pred_quat, dim=-1)
        q2 = F.normalize(gt_quat, dim=-1)
        dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(1e-7, 1.0 - 1e-7)
        return torch.acos(dot).mean()
        
    def configure_optimizers(self):
        """Configure optimizer with warm-up and scheduling"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Period multiplier after restart
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }
        
    def training_step(self, batch, batch_idx):
        images = batch["image_sequence"]
        labels = batch["label"]
        
        predictions, states, uncertainties = self(images, return_all_states=True)
        
        loss, loss_components = self.loss(predictions, labels, states, uncertainties, predictions)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(
            {f"train_{k}": v for k, v in loss_components.items()},
            on_epoch=True
        )
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images = batch["image_sequence"]
        labels = batch["label"]
        
        predictions, states, uncertainties = self(images, return_all_states=True)
        loss, loss_components = self.loss(predictions, labels, states, uncertainties)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict({f"val_{k}": v for k, v in loss_components.items()})
        
        return loss
        
    def test_step(self, batch, batch_idx):
        """Test step with detailed metrics"""
        images = batch["image_sequence"]
        labels = batch["label"]
        
        predictions, states, uncertainties = self(images, return_all_states=True)
        loss, loss_components = self.loss(predictions, labels, states, uncertainties)
        
        # Extract final predictions
        if len(predictions.shape) == 3:
            final_pred = predictions[:, -1]
        else:
            final_pred = predictions
            
        # Calculate additional metrics
        pred_dir = F.normalize(final_pred[:, 0:3], dim=-1)
        gt_dir = F.normalize(labels[:, 0:3], dim=-1)
        
        # Direction angle error in degrees
        cos_sim = torch.sum(pred_dir * gt_dir, dim=-1).clamp(-1.0, 1.0)
        angle_error_deg = torch.rad2deg(torch.acos(cos_sim))
        
        # Distance error
        pred_dist = final_pred[:, 3]
        gt_dist = labels[:, 3]
        dist_error = torch.abs(pred_dist - gt_dist)
        
        # Log all metrics
        self.log("test_loss", loss)
        self.log_dict({
            "test_direction_angle_degrees": angle_error_deg.mean(),
            "test_distance_error": dist_error.mean(),
            **{f"test_{k}": v for k, v in loss_components.items()}
        })
        
        return {
            "test_loss": loss,
            "direction_angle_degrees": angle_error_deg.mean(),
            "distance_error": dist_error.mean(),
            **loss_components
        }
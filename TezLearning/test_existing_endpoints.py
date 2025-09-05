import json
import math
import os
from datetime import datetime

import pytorch_lightning as pl
import torch

# Copy the ImprovedDroneModel class from the other script
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

# Import your model and dataset classes
from dataloader import StateBasedDroneDataset


class ImprovedDroneModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Choose backbone based on config
        if config['backbone'] == 'resnet34':
            backbone = models.resnet34(weights="IMAGENET1K_V1")
            feature_dim = 512
        elif config['backbone'] == 'resnet50':
            backbone = models.resnet50(weights="IMAGENET1K_V1")
            feature_dim = 2048
        elif config['backbone'] == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
            feature_dim = 1280
        
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # Adaptive pooling for different backbones
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Build head with configurable depth
        layers = []
        in_dim = feature_dim
        
        for hidden_dim in config['hidden_dims']:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate'])
            ])
            in_dim = hidden_dim
        
        # Final layer - now outputs 11 values: 3 dir + 1 dist + 4 quat + 3 velocity
        layers.append(nn.Linear(in_dim, 11))
        self.head = nn.Sequential(*layers)
        
        # Loss weights
        self.dir_weight = config['dir_weight']
        self.dist_weight = config['dist_weight']
        self.rot_weight = config['rot_weight']
        self.vel_weight = config.get('vel_weight', 1.0)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features).squeeze(-1).squeeze(-1)
        out = self.head(features)
        
        direction = F.normalize(out[:, :3], dim=-1)
        distance = F.softplus(out[:, 3:4])
        quaternion = F.normalize(out[:, 4:8], dim=-1)
        velocity = out[:, 8:11]  # No normalization for velocity
        
        return torch.cat([direction, distance, quaternion, velocity], dim=-1)
    
    def direction_loss(self, pred_dir, target_dir):
        return 1 - F.cosine_similarity(pred_dir, target_dir, dim=-1).mean()
    
    def distance_loss(self, pred_dist, target_dist):
        if self.config.get('distance_loss_type', 'mse') == 'mse':
            return F.mse_loss(pred_dist, target_dist)
        else:  # huber
            return F.huber_loss(pred_dist, target_dist)
    
    def quaternion_geodesic_loss(self, pred_quat, target_quat):
        dot = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
        dot = torch.clamp(dot, 0.0, 1.0)
        return (2 * torch.acos(dot)).mean()
    
    def velocity_loss(self, pred_vel, target_vel):
        return F.mse_loss(pred_vel, target_vel)
    
    def compute_loss(self, pred, gt):
        pred_dir = pred[:, 0:3]
        pred_dist = pred[:, 3:4]
        pred_quat = pred[:, 4:8]
        pred_vel = pred[:, 8:11]
        
        gt_dir = gt[:, 0:3]
        gt_dist = gt[:, 3:4]
        gt_quat = gt[:, 4:8]
        gt_vel = gt[:, 8:11]
        
        dir_loss = self.direction_loss(pred_dir, gt_dir)
        dist_loss = self.distance_loss(pred_dist, gt_dist)
        rot_loss = self.quaternion_geodesic_loss(pred_quat, gt_quat)
        vel_loss = self.velocity_loss(pred_vel, gt_vel)
        
        total_loss = (self.dir_weight * dir_loss + 
                     self.dist_weight * dist_loss + 
                     self.rot_weight * rot_loss +
                     self.vel_weight * vel_loss)
        
        return total_loss, {
            "dir_loss": dir_loss,
            "dist_loss": dist_loss,
            "rot_loss": rot_loss,
            "vel_loss": vel_loss
        }

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

def create_test_dataset():
    """Create test dataset"""
    dataset = StateBasedDroneDataset(
        images_folder="/home/alp/noetic_ws/TezLearning/data/images",
        csv_path="/home/alp/noetic_ws/TezLearning/data/images/cargo_data.csv"
    )
    
    total_sequences = len(dataset)
    n_train = int(0.7 * total_sequences)
    n_val = int(0.15 * total_sequences)
    
    test_indices = list(range(n_train + n_val, total_sequences))
    test_dataset = Subset(dataset, test_indices)
    
    return test_dataset

def load_existing_results():
    """Load existing experiment results"""
    results = []
    
    # Look for existing result files
    for file in os.listdir('.'):
        if file.startswith('hp_search_results_') and file.endswith('.json'):
            try:
                with open(file, 'r') as f:
                    file_results = json.load(f)
                    results.extend(file_results)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    # Remove duplicates based on experiment_id
    unique_results = {}
    for result in results:
        exp_id = result.get('experiment_id')
        if exp_id is not None:
            # Keep the result with best_val_loss (most recent/complete)
            if exp_id not in unique_results or result.get('best_val_loss', float('inf')) < unique_results[exp_id].get('best_val_loss', float('inf')):
                unique_results[exp_id] = result
    
    return list(unique_results.values())

def test_checkpoint(config, checkpoint_path, experiment_id):
    """Test a single checkpoint"""
    print(f"\n{'='*50}")
    print(f"Testing Experiment {experiment_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Create test dataset and loader
        test_dataset = create_test_dataset()
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=8
        )
        
        # Load model from checkpoint
        model = ImprovedDroneModel.load_from_checkpoint(checkpoint_path, config=config)
        
        # Create trainer for testing
        trainer = pl.Trainer(
            accelerator="gpu",
            precision="16-mixed",
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        
        # Run test
        test_results = trainer.test(model, test_loader)
        test_metrics = test_results[0] if test_results else None
        
        if test_metrics:
            print(f"✓ Test completed successfully")
            print(f"  Test Loss: {test_metrics.get('test_loss', 'N/A'):.6f}")
            print(f"  Direction Error: {test_metrics.get('test_direction_angle_error_deg', 'N/A'):.2f}°")
            print(f"  Distance Error: {test_metrics.get('test_distance_error_m', 'N/A'):.4f}m")
            print(f"  Rotation Error: {test_metrics.get('test_rotation_angle_error_deg', 'N/A'):.2f}°")
        
        return test_metrics
        
    except Exception as e:
        print(f"❌ Test failed for experiment {experiment_id}: {str(e)}")
        return None

def main():
    print("🧪 Testing Existing Checkpoints")
    print("="*60)
    
    # Load existing results
    existing_results = load_existing_results()
    
    if not existing_results:
        print("No existing results found. Make sure you have run the hyperparameter search first.")
        return
    
    print(f"Found {len(existing_results)} existing experiments")
    
    # Test each checkpoint
    updated_results = []
    
    for result in existing_results:
        exp_id = result.get('experiment_id')
        config = result.get('config')
        checkpoint_path = result.get('checkpoint_path')
        
        if not checkpoint_path or not config:
            print(f"⚠️  Skipping experiment {exp_id}: Missing config or checkpoint path")
            updated_results.append(result)
            continue
        
        # Check if test results already exist
        if 'test_results' in result and result['test_results']:
            print(f"⚠️  Experiment {exp_id} already has test results, skipping...")
            updated_results.append(result)
            continue
        
        # Run test
        test_metrics = test_checkpoint(config, checkpoint_path, exp_id)
        
        # Update result with test metrics
        result['test_results'] = test_metrics
        updated_results.append(result)
    
    # Sort by validation loss
    successful_results = [r for r in updated_results if r.get('best_val_loss') != float('inf') and 'error' not in r]
    successful_results.sort(key=lambda x: x.get('best_val_loss', float('inf')))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if successful_results:
        print(f"\nRanking by Validation Loss:")
        print("-" * 100)
        print(f"{'Rank':<4} {'Exp':<3} {'Val Loss':<10} {'Test Loss':<10} {'Dir(°)':<8} {'Dist(m)':<8} {'Rot(°)':<8}")
        print("-" * 100)
        
        for i, result in enumerate(successful_results):
            test_res = result.get('test_results', {})
            val_loss = result.get('best_val_loss', 'N/A')
            test_loss = test_res.get('test_loss', 'N/A') if test_res else 'N/A'
            dir_err = test_res.get('test_direction_angle_error_deg', 'N/A') if test_res else 'N/A'
            dist_err = test_res.get('test_distance_error_m', 'N/A') if test_res else 'N/A'
            rot_err = test_res.get('test_rotation_angle_error_deg', 'N/A') if test_res else 'N/A'
            
            print(f"{i+1:<4} {result.get('experiment_id', 'N/A'):<3} {val_loss:<10.6f} "
                  f"{test_loss:<10.6f} {dir_err:<8.2f} {dist_err:<8.4f} {rot_err:<8.2f}")
        
        # Show best model details
        print(f"\n🏆 BEST MODEL (Experiment {successful_results[0]['experiment_id']}):")
        best_config = successful_results[0]['config']
        best_test = successful_results[0].get('test_results', {})
        
        print(f"Validation Loss: {successful_results[0]['best_val_loss']:.6f}")
        if best_test:
            print(f"Test Loss: {best_test.get('test_loss', 'N/A'):.6f}")
            print(f"Direction Error: {best_test.get('test_direction_angle_error_deg', 'N/A'):.2f}°")
            print(f"Distance Error: {best_test.get('test_distance_error_m', 'N/A'):.4f}m")
            print(f"Rotation Error: {best_test.get('test_rotation_angle_error_deg', 'N/A'):.2f}°")
        
        print(f"\nBest Config:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        
        print(f"\nCheckpoint: {successful_results[0]['checkpoint_path']}")
    
    # Save updated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'complete_results_with_test_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'all_results': updated_results,
            'ranked_results': successful_results,
            'summary': {
                'total_experiments': len(updated_results),
                'successful_experiments': len(successful_results),
                'best_val_loss': successful_results[0]['best_val_loss'] if successful_results else None,
                'best_experiment_id': successful_results[0]['experiment_id'] if successful_results else None
            }
        }, f, indent=2, default=str)
    
    print(f"\n📁 Complete results saved to: {output_file}")

if __name__ == "__main__":
    main()
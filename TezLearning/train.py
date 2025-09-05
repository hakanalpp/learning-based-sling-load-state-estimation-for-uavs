import glob
import os
import random

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataloader import StateBasedDroneDataset
from model import DroneModel

pl.seed_everything(42)

def collect_and_merge_runs(base_path="/home/alp/noetic_ws/TezLearning/data/images"):
    """
    Collect all runs and merge their CSV data, then shuffle completely
    """
    all_data = []
    run_folders = glob.glob(os.path.join(base_path, "run_*"))
    run_folders.sort()  # Ensure consistent ordering
    
    print(f"Found {len(run_folders)} runs: {[os.path.basename(f) for f in run_folders]}")
    
    for run_folder in run_folders:
        csv_path = os.path.join(run_folder, "cargo_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add run identifier to track which run each sample came from
            df['run_id'] = os.path.basename(run_folder)
            
            # CRITICAL: Update frameId to include run folder path
            # The dataloader looks for images using: os.path.join(images_folder, f"{frame_id}.jpg")
            # So we need to prepend the run folder to frameId
            run_name = os.path.basename(run_folder)
            df['frameId'] = df['frameId'].apply(lambda x: f"{run_name}/{x}")
            
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {run_folder}")
        else:
            print(f"Warning: No CSV found in {run_folder}")
    
    if not all_data:
        raise ValueError("No valid run data found!")
    
    # Merge all data
    merged_data = pd.concat(all_data, ignore_index=True)
    print(f"Total merged samples: {len(merged_data)}")
    
    # Shuffle completely
    merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return merged_data, base_path

def create_datasets_with_stratified_split():
    """
    Create datasets with proper stratified splitting across runs
    """
    merged_data, base_path = collect_and_merge_runs()
    
    # Create stratified split to ensure each run is represented in train/val/test
    # Using run_id as stratification key
    train_data, temp_data = train_test_split(
        merged_data, 
        test_size=0.3, 
        stratify=merged_data['run_id'], 
        random_state=42
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        stratify=temp_data['run_id'], 
        random_state=42
    )
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print("Run distribution in splits:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        run_counts = split_data['run_id'].value_counts()
        print(f"  {split_name}: {dict(run_counts)}")
    
    # Save split CSV files for reproducibility
    os.makedirs("./data_splits", exist_ok=True)
    train_data.to_csv("./data_splits/train_data.csv", index=False)
    val_data.to_csv("./data_splits/val_data.csv", index=False)
    test_data.to_csv("./data_splits/test_data.csv", index=False)
    
    # Create datasets using the split data
    train_dataset = StateBasedDroneDataset(
        images_folder=base_path,
        csv_data=train_data  # Pass dataframe directly instead of CSV path
    )
    
    val_dataset = StateBasedDroneDataset(
        images_folder=base_path,
        csv_data=val_data
    )
    
    test_dataset = StateBasedDroneDataset(
        images_folder=base_path,
        csv_data=test_data
    )
    
    return train_dataset, val_dataset, test_dataset



def main(weights_file, test):
    train_dataset, val_dataset, test_dataset = create_datasets_with_stratified_split()
    
    
    # Increased num_workers for larger dataset
    num_workers = min(32, os.cpu_count())  # Use more workers but cap at CPU count
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch more batches
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if test and not os.path.exists(weights_file):
        print("No weights found, cannot run test mode.")
        return

    model = None
    if not os.path.exists(weights_file):
        print("No weights found, training from scratch.")
        model = DroneModel()
    else:
        print("Loading existing weights.")
        model = DroneModel.load_from_checkpoint(weights_file)
        
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=5,  # Keep more checkpoints for larger dataset
            filename="cargo_model-{epoch:02d}-{val_loss:.4f}",
            dirpath="./checkpoints",
            save_last=True,  # Always save the last checkpoint
        ),
        EarlyStopping(
            monitor="val_loss", 
            patience=25,  # Increased patience for larger dataset
            mode="min",
            min_delta=1e-4,  # Require minimum improvement
        ),
        LearningRateMonitor(logging_interval='epoch'),  # Monitor LR changes
    ]

    # Adjusted training parameters for larger dataset
    trainer = pl.Trainer(
        max_epochs=120,  # More epochs for larger dataset
        accelerator="gpu",
        callbacks=callbacks,
        gradient_clip_val=1.0,  # Slightly higher gradient clipping
        precision="16-mixed",
        accumulate_grad_batches=2,  # Gradient accumulation for smaller batches
        val_check_interval=0.5,  # Check validation more frequently
        log_every_n_steps=50,  # Log more frequently
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    if not test:
        trainer.fit(model, train_loader, val_loader)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True,
    )
    trainer.test(model, test_loader)


if __name__ == "__main__":
    # For training
    # main()
    
    # For testing
    main("/home/alp/noetic_ws/TezLearning/checkpoints/cargo_model-epoch=11-val_loss=0.0362.ckpt", True)    # For training

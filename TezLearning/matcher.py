import glob
import os
import shutil
from pathlib import Path

import pandas as pd


def match_images_to_csv(csv_file_path, images_directory, threshold_ms=1, dry_run=False):
    """
    Match images to CSV rows based on Unity timestamps and rename accordingly.
    Modifies the CSV file in place by removing unused rows.
    
    Args:
        csv_file_path: Path to the CSV file
        images_directory: Directory containing the images
        threshold_ms: Threshold in milliseconds for matching (default: 1ms)
        dry_run: If True, only show what would be done without making changes (default: False)
                 In dry run mode, creates 'match_analysis_dryrun.csv' with matching statistics
    """
    
    # Convert milliseconds to ticks (1ms = 10,000 ticks in .NET)
    threshold_ticks = threshold_ms * 10000
    
    if dry_run:
        print("=" * 50)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 50)
    
    print(f"Loading CSV from: {csv_file_path}")
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print(f"Found {len(df)} rows in CSV")
    print(f"CSV timestamp range: {df['frameId'].min()} to {df['frameId'].max()}")
    
    # Get all image files
    image_pattern = os.path.join(images_directory, "*.jpg")
    image_files = glob.glob(image_pattern)
    
    print(f"Found {len(image_files)} images in directory")
    
    # Extract timestamps from image filenames
    image_timestamps = []
    image_paths = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Extract timestamp (remove .jpg extension)
        timestamp_str = filename.replace('.jpg', '')
        try:
            timestamp = int(timestamp_str)
            image_timestamps.append(timestamp)
            image_paths.append(img_path)
        except ValueError:
            print(f"Warning: Could not parse timestamp from {filename}")
    
    print(f"Successfully parsed {len(image_timestamps)} image timestamps")
    if image_timestamps:
        print(f"Image timestamp range: {min(image_timestamps)} to {max(image_timestamps)}")
    
    # Sort images by timestamp for sequential processing
    sorted_images = sorted(zip(image_timestamps, image_paths))
    
    # Track matches and used CSV indices
    matched_csv_indices = []
    csv_index = 0  # Start from first CSV row
    
    # Track matching details for dry run analysis
    match_details = []
    
    # Process each image sequentially
    for img_timestamp, img_path in sorted_images:
        matched = False
        original_filename = os.path.basename(img_path)
        
        # Look for closest match starting from current CSV position
        best_match_idx = None
        best_difference = float('inf')
        
        # Search forward from current position (sequential matching)
        for i in range(csv_index, len(df)):
            csv_timestamp = df.iloc[i]['frameId']
            difference = abs(img_timestamp - csv_timestamp)
            
            if difference <= threshold_ticks:
                if difference < best_difference:
                    best_difference = difference
                    best_match_idx = i
                # Since we're going sequentially, break after finding first good match
                break
            elif csv_timestamp > img_timestamp + threshold_ticks:
                # CSV timestamp is too far ahead, no point checking further
                break
        
        if best_match_idx is not None:
            # Found a match
            csv_timestamp = df.iloc[best_match_idx]['frameId']
            matched_csv_indices.append(best_match_idx)
            
            # Calculate millisecond difference for analysis
            difference_ms = best_difference / 10000  # Convert ticks to milliseconds
            
            # Record match details for dry run analysis
            match_details.append({
                'csv_frame_id': int(csv_timestamp),
                'image_frame_id': img_timestamp,
                'difference_ms': difference_ms
            })
            
            # Rename image to match CSV timestamp (convert to string to avoid scientific notation)
            new_filename = f"{int(csv_timestamp)}.jpg"
            new_path = os.path.join(images_directory, new_filename)
            
            # Only rename if the name is different
            if original_filename != new_filename:
                if dry_run:
                    print(f"[DRY RUN] Would rename: {original_filename} -> {new_filename}")
                else:
                    try:
                        shutil.move(img_path, str(new_path))
                        print(f"Renamed: {original_filename} -> {new_filename}")
                    except Exception as e:
                        print(f"Error renaming {original_filename}: {e}")
            else:
                if dry_run:
                    print(f"[DRY RUN] Already correct name: {original_filename}")
                else:
                    print(f"Already correct name: {original_filename}")
            
            # Update csv_index to continue from next position
            csv_index = best_match_idx + 1
            matched = True
        
        if not matched:
            # No match found, rename as abandoned
            abandoned_filename = f"{img_timestamp}_abandoned.jpg"
            abandoned_path = os.path.join(images_directory, abandoned_filename)
            
            if dry_run:
                print(f"[DRY RUN] Would mark as abandoned: {original_filename} -> {abandoned_filename}")
            else:
                try:
                    shutil.move(img_path, abandoned_path)
                    print(f"No match found: {original_filename} -> {abandoned_filename}")
                except Exception as e:
                    print(f"Error renaming {original_filename}: {e}")
    
    # Remove unused CSV rows (keep only matched rows)
    if matched_csv_indices:
        cleaned_df = df.iloc[matched_csv_indices].copy()
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        print(f"Keeping {len(cleaned_df)} out of {len(df)} CSV rows")
        
        # Save cleaned CSV back to original file
        if dry_run:
            print(f"[DRY RUN] Would overwrite original CSV: {csv_file_path}")
        else:
            cleaned_df.to_csv(csv_file_path, index=False)
            print(f"Original CSV updated: {csv_file_path}")
    else:
        if dry_run:
            print("[DRY RUN] No matches found, original CSV would be left unchanged")
        else:
            print("No matches found, original CSV left unchanged")
    
    # Create match analysis CSV during dry run
    if dry_run and match_details:
        match_analysis_df = pd.DataFrame(match_details)
        analysis_filename = f"match_analysis_dryrun.csv"
        analysis_path = os.path.join(images_directory, analysis_filename)
        match_analysis_df.to_csv(analysis_path, index=False)
        print(f"[DRY RUN] Match analysis saved to: {analysis_path}")
        print(f"[DRY RUN] Average difference: {match_analysis_df['difference_ms'].mean():.3f} ms")
        print(f"[DRY RUN] Max difference: {match_analysis_df['difference_ms'].max():.3f} ms")
        print(f"[DRY RUN] Min difference: {match_analysis_df['difference_ms'].min():.3f} ms")
    
    # Print summary
    total_images = len(image_timestamps)
    matched_images = len(matched_csv_indices)
    abandoned_images = total_images - matched_images
    
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"Total images processed: {total_images}")
    print(f"Images matched: {matched_images}")
    print(f"Images abandoned: {abandoned_images}")
    print(f"CSV rows removed: {len(df) - len(matched_csv_indices) if matched_csv_indices else len(df)}")
    
    if dry_run:
        print("\nNo files or data were actually modified in this dry run.")
        if match_details:
            print("Check 'match_analysis_dryrun.csv' for detailed matching statistics.")

# Example usage
if __name__ == "__main__":
    # Update these paths according to your setup
    csv_file = "/home/alp/noetic_ws/src/simulation/images/run_10/cargo_data.csv"  # Path to your CSV file
    images_dir = "/home/alp/noetic_ws/src/simulation/images/run_10"  # Directory containing images
    
    # First run with dry_run=True to see what would happen
    print("Running dry run to preview changes...")
    match_images_to_csv(csv_file, images_dir, threshold_ms=3, dry_run=True)
    
    # Ask user for confirmation
    user_input = input("\nDo you want to proceed with these changes? (y/N): ")
    
    if user_input.lower() in ['y', 'yes']:
        print("\nRunning actual operations...")
        match_images_to_csv(csv_file, images_dir, threshold_ms=3, dry_run=False)
    else:
        print("Operation cancelled.")
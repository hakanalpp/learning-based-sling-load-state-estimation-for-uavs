import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.transform import Rotation

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TestResultsAnalyzer:
    def __init__(self, csv_path, base_data_path="/home/alp/noetic_ws/TezLearning/data/images"):
        print(f"Loading test results from {csv_path}...")
        start_time = time.time()
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} rows in {time.time() - start_time:.2f}s")
        
        self.base_data_path = base_data_path
        self.original_data_merged = None  # Will store merged original data
        self.process_data()
        
    def load_and_merge_original_data(self):
        """Load and merge all original CSV files once for efficient lookups"""
        print("📂 Loading and merging original data...")
        start_time = time.time()
        
        # Get unique run IDs from test data
        unique_runs = self.df['run_id'].unique()
        all_original_data = []
        
        for run_id in unique_runs:
            csv_path = os.path.join(self.base_data_path, run_id, "cargo_data.csv")
            if os.path.exists(csv_path):
                run_data = pd.read_csv(csv_path)
                run_data['run_id'] = run_id
                # Create lookup key: run_id + timestamp
                run_data['lookup_key'] = run_id + '/' + run_data['frameId'].astype(str)
                all_original_data.append(run_data)
                print(f"  ✓ Loaded {run_id}: {len(run_data)} rows")
            else:
                print(f"  ⚠ Warning: Original data not found at {csv_path}")
        
        if all_original_data:
            self.original_data_merged = pd.concat(all_original_data, ignore_index=True)
            # Set lookup_key as index for fast lookups
            self.original_data_merged.set_index('lookup_key', inplace=True)
            print(f"✓ Merged original data: {len(self.original_data_merged)} total rows in {time.time() - start_time:.2f}s")
        else:
            print("⚠ Warning: No original data found!")
            self.original_data_merged = pd.DataFrame()
        
    def process_data(self):
        """Process the raw data to extract meaningful metrics"""
        print("\n🔄 Starting data processing...")
        total_start = time.time()
        
        # Load and merge original data first
        self.load_and_merge_original_data()
        
        # Parse string representations of lists back to actual lists
        print("📝 Parsing JSON columns...")
        start_time = time.time()
        for col in ['pred_direction', 'gt_direction', 'pred_rotation', 'gt_rotation', 
                   'pred_world_vector', 'gt_world_vector']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        print(f"✓ JSON parsing completed in {time.time() - start_time:.2f}s")
        
        # Calculate angular errors
        print("📐 Calculating direction angle errors...")
        start_time = time.time()
        self.df['direction_angle_error'] = self.df.apply(self.calculate_direction_angle_error, axis=1)
        print(f"✓ Direction angle errors calculated in {time.time() - start_time:.2f}s")
        
        print("📐 Calculating rotation angle errors...")
        start_time = time.time()
        self.df['rotation_angle_error'] = self.df.apply(self.calculate_rotation_angle_error, axis=1)
        print(f"✓ Rotation angle errors calculated in {time.time() - start_time:.2f}s")
        
        # Calculate distance errors in meters
        print("📏 Calculating distance errors...")
        start_time = time.time()
        self.df['distance_error_m'] = abs(self.df['pred_distance'] - self.df['gt_distance'])
        print(f"✓ Distance errors calculated in {time.time() - start_time:.2f}s")
        
        # Calculate predicted cargo position and position mismatch (OPTIMIZED)
        print("📍 Calculating cargo positions and mismatches (vectorized)...")
        start_time = time.time()
        self.calculate_positions_vectorized()
        print(f"✓ Cargo positions and mismatches calculated in {time.time() - start_time:.2f}s")
        
        # Add frame index for plotting
        print("📊 Adding frame indices...")
        start_time = time.time()
        self.df['frame_index'] = range(len(self.df))
        print(f"✓ Frame indices added in {time.time() - start_time:.2f}s")
        
        total_time = time.time() - total_start
        print(f"\n✅ Data processing completed in {total_time:.2f}s total")
        
    def calculate_positions_vectorized(self):
        """Vectorized calculation of cargo positions and mismatches"""
        if self.original_data_merged is None or self.original_data_merged.empty:
            print("⚠ Warning: No original data available, using default values")
            self.df['pred_cargo_pos'] = [[0.0, 0.0, 0.0]] * len(self.df)
            self.df['position_mismatch'] = 0.0
            return
        
        # Merge test data with original data using frameId as key
        merged = self.df.merge(
            self.original_data_merged[['drone_pos_x', 'drone_pos_y', 'drone_pos_z', 
                                     'cargo_pos_x', 'cargo_pos_y', 'cargo_pos_z']],
            left_on='frameId',
            right_index=True,
            how='left'
        )
        
        # Calculate predicted cargo positions vectorized
        drone_positions = merged[['drone_pos_x', 'drone_pos_y', 'drone_pos_z']].values
        pred_world_vectors = np.array(self.df['pred_world_vector'].tolist())
        pred_cargo_positions = drone_positions + pred_world_vectors
        
        # Calculate position mismatches vectorized
        gt_cargo_positions = merged[['cargo_pos_x', 'cargo_pos_y', 'cargo_pos_z']].values
        position_mismatches = np.linalg.norm(pred_cargo_positions - gt_cargo_positions, axis=1)
        
        # Store results
        self.df['pred_cargo_pos'] = pred_cargo_positions.tolist()
        self.df['position_mismatch'] = position_mismatches
        
        # Handle missing data
        missing_mask = merged['drone_pos_x'].isna()
        if missing_mask.any():
            print(f"⚠ Warning: {missing_mask.sum()} frames had no matching original data")
            self.df.loc[missing_mask, 'pred_cargo_pos'] = [[0.0, 0.0, 0.0]] * missing_mask.sum()
            self.df.loc[missing_mask, 'position_mismatch'] = 0.0
        
    def calculate_direction_angle_error(self, row):
        """Calculate angle error between predicted and ground truth direction vectors"""
        pred_dir = np.array(row['pred_direction'])
        gt_dir = np.array(row['gt_direction'])
        
        # Normalize vectors
        pred_dir_norm = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
        gt_dir_norm = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)
        
        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(pred_dir_norm, gt_dir_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_rotation_angle_error(self, row):
        """Calculate angle error between predicted and ground truth quaternions"""
        pred_quat = np.array(row['pred_rotation'])
        gt_quat = np.array(row['gt_rotation'])
        
        # Normalize quaternions
        pred_quat_norm = pred_quat / (np.linalg.norm(pred_quat) + 1e-8)
        gt_quat_norm = gt_quat / (np.linalg.norm(gt_quat) + 1e-8)
        
        # Calculate quaternion difference
        dot_product = np.abs(np.dot(pred_quat_norm, gt_quat_norm))
        dot_product = np.clip(dot_product, 0.0, 1.0)
        angle_rad = 2 * np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def remove_outliers(self, data, column, n_outliers=10):
        """Remove the worst n outliers by value"""
        # Sort by column value and remove the top n_outliers
        threshold = data[column].nlargest(n_outliers).min()
        filtered_data = data[data[column] < threshold]
        outliers_removed = len(data) - len(filtered_data)
        
        return filtered_data, outliers_removed

    def plot_comprehensive_analysis(self, n_outliers_to_remove=50):
        """Publication-ready error distribution plots with smooth density curves"""
        # Set publication style
        plt.style.use('default')  # Clean style for thesis
        
        fig = plt.figure(figsize=(14, 12))  # Taller figure for better visibility of low regions
        
        # Create a 2x2 grid with better spacing
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)  # More vertical spacing
        
        # Define professional color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Remove outliers for better visualization
        df_clean_dir, dir_outliers = self.remove_outliers(self.df, 'direction_angle_error', n_outliers_to_remove)
        df_clean_rot, rot_outliers = self.remove_outliers(self.df, 'rotation_angle_error', n_outliers_to_remove)
        df_clean_dist, dist_outliers = self.remove_outliers(self.df, 'distance_error_m', n_outliers_to_remove)
        df_clean_pos, pos_outliers = self.remove_outliers(self.df, 'position_mismatch', n_outliers_to_remove)
        
        # Calculate statistics for annotations
        dir_mean = df_clean_dir['direction_angle_error'].mean()
        rot_mean = df_clean_rot['rotation_angle_error'].mean()
        dist_mean = df_clean_dist['distance_error_m'].mean()
        pos_mean = df_clean_pos['position_mismatch'].mean()
        
        # 1. Direction angle error distribution
        ax1 = fig.add_subplot(gs[0, 0])
        data = df_clean_dir['direction_angle_error']
        counts, bins = np.histogram(data, bins=40, density=True)
        percentages = counts * np.diff(bins) * 100
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        from scipy.interpolate import make_interp_spline
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 150)
        spl = make_interp_spline(bin_centers, percentages, k=2)
        y_smooth = np.maximum(spl(x_smooth), 0)
        
        ax1.fill_between(x_smooth, y_smooth, alpha=0.7, color=colors[0], label='Error Distribution')
        ax1.plot(x_smooth, y_smooth, color=colors[0], linewidth=0.8, alpha=0.9)  # THINNER LINE
        ax1.axvline(x=dir_mean, color='red', linestyle='--', alpha=0.8, linewidth=1.2, 
                   label=f'Mean: {dir_mean:.2f}°')
        
        # Add performance statistic as third legend item
        low_error_pct = (data <= 1.0).mean() * 100
        ax1.plot([], [], ' ', label=f'{low_error_pct:.1f}% ≤ 1.0°')
        
        ax1.set_title('(a) Direction Error', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xlabel('Angular Error (degrees)', fontsize=11)
        ax1.set_ylabel('Density (%)', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.legend(fontsize=9, loc='upper right')
        ax1.tick_params(labelsize=10)
        
        # 2. Rotation angle error distribution
        ax2 = fig.add_subplot(gs[0, 1])
        data = df_clean_rot['rotation_angle_error']
        counts, bins = np.histogram(data, bins=40, density=True)
        percentages = counts * np.diff(bins) * 100
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 150)
        spl = make_interp_spline(bin_centers, percentages, k=2)
        y_smooth = np.maximum(spl(x_smooth), 0)
        
        ax2.fill_between(x_smooth, y_smooth, alpha=0.7, color=colors[1], label='Error Distribution')
        ax2.plot(x_smooth, y_smooth, color=colors[1], linewidth=0.8, alpha=0.9)  # THINNER LINE
        ax2.axvline(x=rot_mean, color='red', linestyle='--', alpha=0.8, linewidth=1.2,
                   label=f'Mean: {rot_mean:.2f}°')
        
        low_error_pct = (data <= 2.0).mean() * 100
        ax2.plot([], [], ' ', label=f'{low_error_pct:.1f}% ≤ 2.0°')
        
        ax2.set_title('(b) Rotation Error', fontsize=12, fontweight='bold', pad=15)
        ax2.set_xlabel('Angular Error (degrees)', fontsize=11)
        ax2.set_ylabel('Density (%)', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.tick_params(labelsize=10)
        
        # 3. Distance error distribution with higher resolution
        ax3 = fig.add_subplot(gs[1, 0])
        data = df_clean_dist['distance_error_m']
        counts, bins = np.histogram(data, bins=60, density=True)
        percentages = counts * np.diff(bins) * 100
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 200)
        spl = make_interp_spline(bin_centers, percentages, k=2)
        y_smooth = np.maximum(spl(x_smooth), 0)
        
        ax3.fill_between(x_smooth, y_smooth, alpha=0.7, color=colors[2], label='Error Distribution')
        ax3.plot(x_smooth, y_smooth, color=colors[2], linewidth=0.8, alpha=0.9)  # THINNER LINE
        ax3.axvline(x=dist_mean, color='red', linestyle='--', alpha=0.8, linewidth=1.2,
                   label=f'Mean: {dist_mean:.3f}m')
        
        low_error_pct = (data <= 0.01).mean() * 100
        ax3.plot([], [], ' ', label=f'{low_error_pct:.1f}% ≤ 0.01m')
        
        ax3.set_title('(c) Distance Error', fontsize=12, fontweight='bold', pad=15)
        ax3.set_xlabel('Distance Error (meters)', fontsize=11)
        ax3.set_ylabel('Density (%)', fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax3.legend(fontsize=9, loc='upper right')
        ax3.tick_params(labelsize=10)
        
        # Set more detailed x-axis ticks for distance plot
        ax3.set_xticks(np.arange(0, data.max() + 0.005, 0.005))
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Position mismatch error distribution
        ax4 = fig.add_subplot(gs[1, 1])
        data = df_clean_pos['position_mismatch']
        counts, bins = np.histogram(data, bins=40, density=True)
        percentages = counts * np.diff(bins) * 100
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 150)
        spl = make_interp_spline(bin_centers, percentages, k=2)
        y_smooth = np.maximum(spl(x_smooth), 0)
        
        ax4.fill_between(x_smooth, y_smooth, alpha=0.7, color=colors[3], label='Error Distribution')
        ax4.plot(x_smooth, y_smooth, color=colors[3], linewidth=0.8, alpha=0.9)  # THINNER LINE
        ax4.axvline(x=pos_mean, color='red', linestyle='--', alpha=0.8, linewidth=1.2,
                   label=f'Mean: {pos_mean:.3f}m')
        
        low_error_pct = (data <= 0.02).mean() * 100
        ax4.plot([], [], ' ', label=f'{low_error_pct:.1f}% ≤ 0.02m')
        
        ax4.set_title('(d) 3D Position Error', fontsize=12, fontweight='bold', pad=15)
        ax4.set_xlabel('3D Position Error (meters)', fontsize=11)
        ax4.set_ylabel('Density (%)', fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax4.legend(fontsize=9, loc='upper right')
        ax4.tick_params(labelsize=10)
        
        # Print info about high error samples to console (simple way to show rarity)
        print(f"\n📊 High Error Analysis:")
        high_dir = (df_clean_dir['direction_angle_error'] > 3.0).sum()
        high_rot = (df_clean_rot['rotation_angle_error'] > 5.0).sum()
        high_dist = (df_clean_dist['distance_error_m'] > 0.04).sum()
        high_pos = (df_clean_pos['position_mismatch'] > 0.08).sum()
        
        print(f"  Direction errors > 3°: {high_dir} samples ({high_dir/len(df_clean_dir)*100:.2f}%)")
        print(f"  Rotation errors > 5°: {high_rot} samples ({high_rot/len(df_clean_rot)*100:.2f}%)")
        print(f"  Distance errors > 4cm: {high_dist} samples ({high_dist/len(df_clean_dist)*100:.2f}%)")
        print(f"  Position errors > 8cm: {high_pos} samples ({high_pos/len(df_clean_pos)*100:.2f}%)")
        
        plt.savefig('comprehensive_analysis.png', dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig('comprehensive_analysis.pdf', dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("✓ Saved publication-ready smooth density plots:")
        print("  - comprehensive_analysis.png (600 DPI)")
        print("  - comprehensive_analysis.pdf (for LaTeX)")
    
    def generate_statistics_table(self):
        """Generate comprehensive statistics table"""
        metrics = ['Direction Error (degrees)', 'Rotation Error (degrees)', 
                  'Distance Error (meters)', 'Position Mismatch (meters)']
        columns = ['direction_angle_error', 'rotation_angle_error', 'distance_error_m', 
                  'position_mismatch']
        
        stats = {
            'Metric': metrics,
            'Mean': [],
            'Median': [],
            'Std Dev': [],
            'Min': [],
            'Max': [],
            '95th Percentile': []
        }
        
        for col in columns:
            stats['Mean'].append(self.df[col].mean())
            stats['Median'].append(self.df[col].median())
            stats['Std Dev'].append(self.df[col].std())
            stats['Min'].append(self.df[col].min())
            stats['Max'].append(self.df[col].max())
            stats['95th Percentile'].append(self.df[col].quantile(0.95))
        
        stats_df = pd.DataFrame(stats)
        
        # Round to appropriate decimal places
        for col in ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '95th Percentile']:
            stats_df[col] = stats_df[col].round(4)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS STATISTICS")
        print("="*80)
        print(stats_df.to_string(index=False))
        print("="*80)
        
        return stats_df
    
    def run_complete_analysis(self, n_outliers_to_remove=40):
        """Run the complete analysis pipeline"""
        print("\n🚀 Starting comprehensive test results analysis...")
        analysis_start = time.time()
        
        # Generate statistics
        print("\n📊 Generating statistics table...")
        stats_start = time.time()
        stats_df = self.generate_statistics_table()
        print(f"✓ Statistics generated in {time.time() - stats_start:.2f}s")
        
        # Generate the single comprehensive plot
        print(f"\n📈 Generating comprehensive analysis plot (removing {n_outliers_to_remove} worst outliers per metric)...")
        plot_start = time.time()
        self.plot_comprehensive_analysis(n_outliers_to_remove)
        print(f"✓ Plot generated in {time.time() - plot_start:.2f}s")
        
        # Save statistics to CSV
        print("\n💾 Saving statistics to CSV...")
        save_start = time.time()
        stats_df.to_csv('test_statistics.csv', index=False)
        print(f"✓ Statistics saved in {time.time() - save_start:.2f}s")
        
        total_analysis_time = time.time() - analysis_start
        print(f"\n🎉 Analysis complete in {total_analysis_time:.2f}s total!")
        print("Generated files:")
        print("- comprehensive_analysis.png (single plot with all key metrics)")
        print("- test_statistics.csv (detailed statistics)")
        
        return stats_df


# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your test results CSV and base path to run folders
    analyzer = TestResultsAnalyzer('test_results.csv', '/home/alp/noetic_ws/TezLearning/data/images')
    
    # Run complete analysis - you can adjust the number of outliers to remove
    stats = analyzer.run_complete_analysis(n_outliers_to_remove=40)
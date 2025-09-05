"""
Waypoint Path Generator
Creates waypoints with bell curve distance distribution.
Most distances ~10m, occasionally up to 100m.
"""

import os

import numpy as np

INPUT_FILE = "og_mission.yaml"  # Change this to your input file name
OUTPUT_PREFIX = "out"   # Output files will be named: route.yaml, route_reverse.yaml

MEAN_DISTANCE = 15.0     # Peak of bell curve (most common distance)
STD_DEVIATION = 12.0     # Spread of bell curve (higher = more variation)
MIN_DISTANCE = 10.0      # Minimum distance between waypoints
MAX_DISTANCE = 100.0     # Maximum distance between waypoints
EXACT_WAYPOINTS = 250    # Exact number of waypoints (will be exactly this many)

GENERATE_REVERSE = True  # Set to False if you don't want reverse route
SHOW_STATISTICS = True   # Set to False to hide statistics


def parse_waypoint_file(file_path):
    """Parse the waypoint file and extract route data."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    lines = content.strip().split('\n')
    
    settings = {}
    for line in lines[:5]:
        if ':' in line:
            key, value = line.split(':', 1)
            settings[key.strip()] = float(value.strip())
    
    initial_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'initial:':
            initial_start = i
            break
    
    initial_point = [0, 0, 0]  # [x, z, y]
    for i in range(initial_start + 1, initial_start + 4):
        line = lines[i].strip()
        if 'x:' in line:
            initial_point[0] = int(line.split(':')[1].strip())
        elif 'z:' in line:
            initial_point[1] = int(line.split(':')[1].strip())
        elif 'y:' in line:
            initial_point[2] = int(line.split(':')[1].strip())
    
    waypoints_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'waypoints:':
            waypoints_start = i
            break
    
    waypoints = []
    for i in range(waypoints_start + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith('- { x:'):
            # Parse: - { x: -450, z: 40, y: 269 }
            coords = line.replace('- { x:', '').replace('}', '').strip()
            parts = coords.split(',')
            x = int(parts[0].strip())
            z = int(parts[1].split(':')[1].strip())
            y = int(parts[2].split(':')[1].strip())
            waypoints.append([x, z, y])  # Keep original x, z, y order
    
    closed_loop = [initial_point] + waypoints + [initial_point]
    
    formatted_settings = {
        'speed': settings.get('speed', 10.0),
        'initial_hover_time': settings.get('initial_hover_time', 5.0),
        'waypoint_hover_time': settings.get('waypoint_hover_time', 3.0),
        'waypoint_threshold': settings.get('waypoint_threshold', 0.5),
        'acceleration_limit': settings.get('acceleration_limit', 1.0)
    }
    
    return closed_loop, initial_point, formatted_settings

def calculate_path_distances(points):
    """Calculate cumulative distances along the path."""
    cumulative_distances = [0]
    total_distance = 0
    
    for i in range(1, len(points)):
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        segment_distance = np.linalg.norm(p2 - p1)
        total_distance += segment_distance
        cumulative_distances.append(total_distance)
    
    return cumulative_distances, total_distance

def generate_bell_curve_distances(total_distance, target_waypoints):
    """Generate exactly (target_waypoints-1) distances following bell curve, scaled to total_distance."""
    num_distances = target_waypoints - 1  # 249 distances for 250 waypoints
    
    distances = np.random.normal(MEAN_DISTANCE, STD_DEVIATION, num_distances)
    
    distances = np.clip(distances, MIN_DISTANCE, MAX_DISTANCE)
    
    current_sum = np.sum(distances)
    scale_factor = total_distance / current_sum
    scaled_distances = distances * scale_factor
    
    final_sum = np.sum(scaled_distances)
    
    print(f"Generated exactly {len(scaled_distances)} bell curve distances")
    print(f"Original sum: {current_sum:.1f}m, Scaled sum: {final_sum:.1f}m (target: {total_distance:.1f}m)")
    print(f"Scale factor: {scale_factor:.3f}")
    print(f"Distance stats after scaling: mean={np.mean(scaled_distances):.1f}, "
          f"std={np.std(scaled_distances):.1f}, "
          f"min={min(scaled_distances):.1f}, max={max(scaled_distances):.1f}")
    
    return scaled_distances

def interpolate_point_on_segment(p1, p2, t):
    """Interpolate a point between p1 and p2 at parameter t (0 to 1)."""
    p1, p2 = np.array(p1), np.array(p2)
    interpolated = p1 + t * (p2 - p1)
    return [int(round(interpolated[0])), int(round(interpolated[1])), int(round(interpolated[2]))]

def generate_bell_curve_waypoints(original_points, target_waypoints):
    """Generate exactly target_waypoints with bell curve distance distribution."""
    cumulative_distances, total_distance = calculate_path_distances(original_points)
    
    print(f"Total path distance: {total_distance:.2f}")
    
    bell_distances = generate_bell_curve_distances(total_distance, target_waypoints)
    
    new_waypoints = []
    current_distance = 0
    
    new_waypoints.append(original_points[0][:])  # Copy the starting point
    
    for i, spacing in enumerate(bell_distances):
        current_distance += spacing
        target_distance = current_distance
        
        segment_index = 0
        for j in range(len(cumulative_distances) - 1):
            if cumulative_distances[j] <= target_distance <= cumulative_distances[j + 1]:
                segment_index = j
                break
        
        segment_start_distance = cumulative_distances[segment_index]
        segment_end_distance = cumulative_distances[segment_index + 1]
        segment_length = segment_end_distance - segment_start_distance
        
        if segment_length == 0:
            new_point = original_points[segment_index][:]  # Copy the point
        else:
            t = (target_distance - segment_start_distance) / segment_length
            
            p1 = original_points[segment_index]
            p2 = original_points[segment_index + 1]
            new_point = interpolate_point_on_segment(p1, p2, t)
        
        new_waypoints.append(new_point)
    
    print(f"Generated exactly {len(new_waypoints)} waypoints")
    
    start_point = np.array(new_waypoints[0])
    end_point = np.array(new_waypoints[-1])
    closure_distance = np.linalg.norm(end_point - start_point)
    print(f"Loop closure distance: {closure_distance:.2f}m")
    
    return new_waypoints

def create_reverse_waypoints(waypoints):
    """Create reverse order waypoints (same starting point, opposite direction)."""
    return [waypoints[0]] + waypoints[1:][::-1]

def save_waypoint_file(waypoints, settings, output_path):
    """Save waypoints to the same format as input (x, z, y order) with distance comments."""
    with open(output_path, 'w') as file:
        file.write(f"speed: {settings['speed']}\n")
        file.write(f"initial_hover_time: {settings['initial_hover_time']}\n")
        file.write(f"waypoint_hover_time: {settings['waypoint_hover_time']}\n")
        file.write(f"waypoint_threshold: {settings['waypoint_threshold']}\n")
        file.write(f"acceleration_limit: {settings['acceleration_limit']}\n\n")
        
        initial = waypoints[0]
        file.write("initial:\n")
        file.write(f"  x: {initial[0]}\n")   # x stays x
        file.write(f"  z: {initial[1]}\n")   # z stays z  
        file.write(f"  y: {initial[2]}\n\n") # y stays y
        
        file.write("waypoints:\n")
        
        previous_point = waypoints[0]  # Start from initial position
        for i, wp in enumerate(waypoints[1:], 1):
            p1 = np.array(previous_point)
            p2 = np.array(wp)
            distance = np.linalg.norm(p2 - p1)
            
            file.write(f"  - {{ x: {wp[0]}, z: {wp[1]}, y: {wp[2]} }}  # {distance:.1f}m from previous\n")
            
            previous_point = wp

def calculate_segment_distances(points):
    """Calculate distances between consecutive points."""
    distances = []
    for i in range(len(points) - 1):  # Don't wrap around for linear analysis
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        dist = np.linalg.norm(p2 - p1)
        distances.append(dist)
    return distances

def print_statistics(points, label):
    """Print statistics about the waypoint spacing."""
    distances = calculate_segment_distances(points)
    if distances:
        print(f"\n{label} Statistics:")
        print(f"  Total waypoints: {len(points)}")
        print(f"  Total segments: {len(distances)}")
        print(f"  Total distance: {sum(distances):.2f}")
        print(f"  Average distance: {np.mean(distances):.2f}")
        print(f"  Min distance: {min(distances):.2f}")
        print(f"  Max distance: {max(distances):.2f}")
        print(f"  Std deviation: {np.std(distances):.2f}")
        
        # Count distances in different ranges
        range_10_20 = sum(1 for d in distances if 10 <= d <= 20)
        range_20_50 = sum(1 for d in distances if 20 < d <= 50)
        range_50_100 = sum(1 for d in distances if 50 < d <= 100)
        
        total = len(distances)
        print(f"  10-20m range: {range_10_20}/{total} ({range_10_20/total*100:.1f}%)")
        print(f"  20-50m range: {range_20_50}/{total} ({range_20_50/total*100:.1f}%)")
        print(f"  50-100m range: {range_50_100}/{total} ({range_50_100/total*100:.1f}%)")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found")
        print("Please update INPUT_FILE variable in the script")
        return
    
    print(f"Reading waypoints from: {INPUT_FILE}")
    try:
        original_loop, initial_point, settings = parse_waypoint_file(INPUT_FILE)
    except Exception as e:
        print(f"Error parsing input file: {e}")
        return
    
    print(f"Loaded {len(original_loop)} points in closed loop")
    print(f"Initial point: x={initial_point[0]}, z={initial_point[1]}, y={initial_point[2]}")
    
    print(f"\nBell curve settings:")
    print(f"  Mean distance: {MEAN_DISTANCE}m")
    print(f"  Std deviation: {STD_DEVIATION}m") 
    print(f"  Min distance: {MIN_DISTANCE}m")
    print(f"  Max distance: {MAX_DISTANCE}m")
    
    print(f"\nGenerating exactly {EXACT_WAYPOINTS} waypoints with bell curve distance distribution...")
    new_waypoints = generate_bell_curve_waypoints(original_loop, EXACT_WAYPOINTS)
    
    print(f"Generated {len(new_waypoints)} waypoints")
    print(f"First waypoint: x={new_waypoints[0][0]}, z={new_waypoints[0][1]}, y={new_waypoints[0][2]}")
    print(f"Last waypoint: x={new_waypoints[-1][0]}, z={new_waypoints[-1][1]}, y={new_waypoints[-1][2]}")
    
    forward_file = f"{OUTPUT_PREFIX}.yaml"
    save_waypoint_file(new_waypoints, settings, forward_file)
    print(f"Saved forward route: {forward_file}")
    
    if GENERATE_REVERSE:
        reverse_waypoints = create_reverse_waypoints(new_waypoints)
        reverse_file = f"{OUTPUT_PREFIX}_reverse.yaml"
        save_waypoint_file(reverse_waypoints, settings, reverse_file)
        print(f"Saved reverse route: {reverse_file}")
    
    if SHOW_STATISTICS:
        print_statistics(new_waypoints, "Bell Curve Forward")
        if GENERATE_REVERSE:
            print_statistics(reverse_waypoints, "Bell Curve Reverse")

if __name__ == "__main__":
    main()
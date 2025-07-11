import math
from typing import Dict, List, Tuple

import yaml


def calculate_distance(point1: Dict, point2: Dict) -> float:
    """Calculate 3D Euclidean distance between two points."""
    dx = point2['x'] - point1['x']
    dy = point2['y'] - point1['y']
    dz = point2['z'] - point1['z']
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def interpolate_point(point1: Dict, point2: Dict, ratio: float) -> Dict:
    """Interpolate a point between two points at given ratio (0-1)."""
    return {
        'x': round(point1['x'] + (point2['x'] - point1['x']) * ratio),
        'y': round(point1['y'] + (point2['y'] - point1['y']) * ratio),
        'z': round(point1['z'] + (point2['z'] - point1['z']) * ratio)
    }

def analyze_and_optimize_waypoints(yaml_file: str, distance_threshold: float = 100.0, add_waypoints: bool = False) -> None:
    """
    Analyze waypoint distances and optionally add intermediate waypoints.
    
    Args:
        yaml_file: Path to the YAML file
        distance_threshold: Maximum allowed distance between consecutive waypoints
        add_waypoints: If True, adds waypoints; if False, just reports analysis
    """
    
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    
    waypoints = data['waypoints']
    initial_point = data['initial']
    
    # Include initial point in the analysis
    all_points = [initial_point] + waypoints
    
    total_waypoints_to_add = 0
    segments_needing_waypoints = []
    new_waypoints = []
    
    # Analyze each segment
    for i in range(len(all_points) - 1):
        current_point = all_points[i]
        next_point = all_points[i + 1]
        distance = calculate_distance(current_point, next_point)
        
        if distance > distance_threshold:
            # Calculate how many intermediate waypoints we need
            num_segments_needed = math.ceil(distance / distance_threshold)
            waypoints_to_add = num_segments_needed - 1
            
            total_waypoints_to_add += waypoints_to_add
            segments_needing_waypoints.append({
                'segment': i,
                'from': current_point,
                'to': next_point,
                'distance': distance,
                'waypoints_to_add': waypoints_to_add
            })
            
            # Generate intermediate waypoints if requested
            if add_waypoints:
                segment_waypoints = []
                for j in range(1, num_segments_needed):
                    ratio = j / num_segments_needed
                    intermediate_point = interpolate_point(current_point, next_point, ratio)
                    segment_waypoints.append(intermediate_point)
                new_waypoints.append({
                    'after_index': i - 1 if i > 0 else 'initial',  # Account for initial point
                    'waypoints': segment_waypoints
                })
    
    # Report analysis
    print(f"=== Waypoint Distance Analysis ===")
    print(f"Distance threshold: {distance_threshold}")
    print(f"Total waypoints in file: {len(waypoints)}")
    print(f"Segments exceeding threshold: {len(segments_needing_waypoints)}")
    print(f"Total waypoints to add: {total_waypoints_to_add}")
    print()
    
    if segments_needing_waypoints:
        print("Segments needing intermediate waypoints:")
        for segment in segments_needing_waypoints:
            if segment['segment'] == 0:
                print(f"  Initial -> Waypoint 1: distance {segment['distance']:.1f}, adding {segment['waypoints_to_add']} waypoint(s)")
            else:
                print(f"  Waypoint {segment['segment']} -> {segment['segment'] + 1}: distance {segment['distance']:.1f}, adding {segment['waypoints_to_add']} waypoint(s)")
    
    # Add waypoints if requested
    if add_waypoints and total_waypoints_to_add > 0:
        print(f"\n=== Adding {total_waypoints_to_add} waypoints ===")
        
        # Build new waypoint list
        updated_waypoints = []
        waypoint_index = 0
        
        for new_waypoint_info in new_waypoints:
            # Add original waypoints up to insertion point
            while waypoint_index <= new_waypoint_info['after_index'] and waypoint_index < len(waypoints):
                if new_waypoint_info['after_index'] != 'initial':
                    updated_waypoints.append(waypoints[waypoint_index])
                waypoint_index += 1
            
            # Add new intermediate waypoints
            updated_waypoints.extend(new_waypoint_info['waypoints'])
        
        # Add remaining original waypoints
        while waypoint_index < len(waypoints):
            updated_waypoints.append(waypoints[waypoint_index])
            waypoint_index += 1
        
        # Update the data structure
        data['waypoints'] = updated_waypoints
        
        # Write back to file with preserved formatting
        output_file = yaml_file.replace('.yaml', '_optimized.yaml').replace('.yml', '_optimized.yml')
        if output_file == yaml_file:  # If no extension was replaced
            output_file = yaml_file + '_optimized'
        
        # Write with preserved structure
        with open(output_file, 'w') as file:
            # Write the header parameters
            file.write(f"speed: {data['speed']}\n")
            file.write(f"initial_hover_time: {data['initial_hover_time']}\n")
            file.write(f"waypoint_hover_time: {data['waypoint_hover_time']}\n")
            file.write(f"waypoint_threshold: {data['waypoint_threshold']}\n")
            file.write(f"acceleration_limit: {data['acceleration_limit']}\n")
            file.write("\n")
            
            # Write initial point
            initial = data['initial']
            file.write("initial:\n")
            file.write(f"  x: {initial['x']}\n")
            file.write(f"  z: {initial['z']}\n")
            file.write(f"  y: {initial['y']}\n")
            file.write("\n")
            
            # Write waypoints in original inline format
            file.write("waypoints:\n")
            for waypoint in data['waypoints']:
                file.write(f"  - {{ x: {waypoint['x']}, z: {waypoint['z']}, y: {waypoint['y']} }}\n")
        
        print(f"Updated file saved as: {output_file}")
        print(f"New total waypoints: {len(updated_waypoints)}")
    
    elif not add_waypoints and total_waypoints_to_add > 0:
        print(f"\nTo add these waypoints, run with add_waypoints=True")
    
    elif total_waypoints_to_add == 0:
        print("\nNo waypoints need to be added - all distances are within threshold!")

# Example usage
if __name__ == "__main__":
    # Example with your file - just analysis
    print("=== ANALYSIS MODE ===")
    analyze_and_optimize_waypoints('/home/alp/noetic_ws/src/controller_pkg/mission.yaml', distance_threshold=60.0, add_waypoints=True)
    

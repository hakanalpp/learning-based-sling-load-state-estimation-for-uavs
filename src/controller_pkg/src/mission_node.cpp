#include <cmath>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sstream>
#include <std_msgs/String.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <trajectory_msgs/MultiDOFJointTrajectoryPoint.h>
#include <vector>
#include <yaml-cpp/yaml.h>

#define PI M_PI

struct Waypoint {
  double x;
  double y;
  double z;
};

class missionNode {
  ros::NodeHandle nh;
  ros::NodeHandle nh_priv = ros::NodeHandle("~");
  ros::Rate loop_rate = ros::Rate(500);

  ros::Publisher desired_state;
  ros::Subscriber current_state_sub;

  double initial_hover_time;
  double waypoint_hover_time;
  double cruise_speed;
  double acceleration_limit;

  tf::Vector3 initial_position = tf::Vector3(0, 0, 0);
  std::vector<Waypoint> waypoints;

  ros::Time start_time = ros::Time::now();

  tf::Transform desired_pose;
  tf::Vector3 current_position;
  tf::Vector3 current_velocity = tf::Vector3(0, 0, 0);

  geometry_msgs::Twist velocity;
  geometry_msgs::Twist acceleration;

  YAML::Node mission_config;

  tf::TransformBroadcaster br;

  int current_wp_index = 0;

public:
  missionNode() {
    desired_state = nh.advertise<trajectory_msgs::MultiDOFJointTrajectoryPoint>(
        "desired_state", 1);
    current_state_sub = nh.subscribe("current_state_est", 1,
                                     &missionNode::currentStateCallback, this);
    loadMission();
    current_position = initial_position;
    startMission();
  }

private:
  void startMission() {
    ROS_INFO("Hovering for %.2f seconds...", initial_hover_time);

    double starting_time = missionTime();
    while (ros::ok() && missionTime() - starting_time < initial_hover_time) {
      reset();
      startHovering();
      publish();
      ros::spinOnce();
      loop_rate.sleep();
    }

    current_wp_index = 0;
    for (; current_wp_index < (int)waypoints.size(); current_wp_index++) {
      goToWaypoint();
    }

    ROS_INFO("Mission complete. Holding final waypoint.");
    while (ros::ok()) {
      reset();
      holdLastWaypoint();
      publish();
      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  void goToWaypoint() {
    Waypoint wp = waypoints[current_wp_index];
    tf::Vector3 target = tf::Vector3(wp.x, wp.y, wp.z);
    ROS_INFO("Going to waypoint %d: [%.2f, %.2f, %.2f]", current_wp_index,
             target.x(), target.y(), target.z());
    
    tf::Vector3 start_position = current_position;
    tf::Vector3 start_velocity = current_velocity;
    
    double dt = 1.0 / 500.0;  // Based on 500Hz loop rate
    bool reached_flag = false;
    double hovering_start_time = 0;
    
    // State variables for smooth trajectory
    tf::Vector3 desired_position = start_position;
    tf::Vector3 desired_velocity = start_velocity;
    
    while (ros::ok()) {
        // Calculate remaining distance to target
        tf::Vector3 to_target = target - desired_position;
        double remaining_distance = to_target.length();
        
        // Define a small threshold for "reached"
        double reached_threshold = 0.1; // 10cm
        
        if (remaining_distance < reached_threshold) {
            // We've reached the waypoint
            if (!reached_flag) {
                ROS_INFO("Reached waypoint %d", current_wp_index);
                reached_flag = true;
                hovering_start_time = missionTime();
            }
            
            // Hover at the waypoint
            desired_position = target;
            desired_velocity = tf::Vector3(0, 0, 0);
            
            if (missionTime() - hovering_start_time >= waypoint_hover_time) {
                // Done hovering, move to next waypoint
                current_velocity = tf::Vector3(0, 0, 0);
                break;
            }
        } else {
            // Still traveling to waypoint
            tf::Vector3 direction_to_target = to_target.normalized();
            
            // Calculate current speed
            double current_speed = desired_velocity.length();
            
            // Calculate stopping distance: d = v²/(2a)
            double stopping_distance = (current_speed * current_speed) / (2.0 * acceleration_limit);
            
            // Determine desired speed based on remaining distance
            double desired_speed;
            if (remaining_distance > stopping_distance) {
                // Far enough to accelerate or maintain cruise speed
                desired_speed = cruise_speed;
            } else {
                // Need to decelerate - calculate speed for smooth stop
                // v = sqrt(2 * a * d)
                desired_speed = std::sqrt(2.0 * acceleration_limit * remaining_distance);
                desired_speed = std::min(desired_speed, current_speed); // Never accelerate during decel
            }
            
            // Calculate required acceleration to reach desired speed
            double speed_change = desired_speed - current_speed;
            double required_accel = speed_change / dt;
            
            // Limit acceleration to acceleration_limit
            double applied_accel;
            if (std::abs(required_accel) > acceleration_limit) {
                applied_accel = (required_accel > 0) ? acceleration_limit : -acceleration_limit;
            } else {
                applied_accel = required_accel;
            }
            
            // Update speed with limited acceleration
            double new_speed = current_speed + applied_accel * dt;
            new_speed = std::max(0.0, std::min(new_speed, cruise_speed));
            
            // Update velocity vector
            desired_velocity = direction_to_target * new_speed;
            
            // Calculate actual acceleration vector for output
            tf::Vector3 accel_vector = direction_to_target * applied_accel;
            
            // Update position
            desired_position += desired_velocity * dt;
            
            // Store current velocity for next iteration
            current_velocity = desired_velocity;
            
            // Set acceleration output
            acceleration.linear.x = accel_vector.x();
            acceleration.linear.y = accel_vector.y();
            acceleration.linear.z = accel_vector.z();
            
            // Update yaw to point in direction of motion
            if (new_speed > 0.1) { // Only update yaw if moving
                tf::Quaternion q;
                q.setRPY(0, 0, atan2(direction_to_target.y(), direction_to_target.x()));
                desired_pose.setRotation(q);
            }
        }
        
        // Update desired pose
        desired_pose.setOrigin(desired_position);
        
        // Set velocity output
        velocity.linear.x = desired_velocity.x();
        velocity.linear.y = desired_velocity.y();
        velocity.linear.z = desired_velocity.z();
        
        publish();
        ros::spinOnce();
        loop_rate.sleep();
    }
  }

  void holdLastWaypoint() {
    Waypoint wp = waypoints.back();
    tf::Vector3 target = tf::Vector3(wp.x, wp.y, wp.z);
    desired_pose.setOrigin(target);
    velocity = geometry_msgs::Twist();
    acceleration = geometry_msgs::Twist();
  }

  void reset() {
    desired_pose = tf::Transform(tf::Transform::getIdentity());
    velocity = geometry_msgs::Twist();
    acceleration = geometry_msgs::Twist();
  }

  void startHovering() {
    desired_pose.setOrigin(initial_position);
    tf::Quaternion q;
    q.setRPY(0, 0, PI / 4);
    desired_pose.setRotation(q);
  }

  void publish() {
    trajectory_msgs::MultiDOFJointTrajectoryPoint msg;
    msg.transforms.resize(1);
    msg.transforms[0].translation.x = desired_pose.getOrigin().x();
    msg.transforms[0].translation.y = desired_pose.getOrigin().y();
    msg.transforms[0].translation.z = desired_pose.getOrigin().z();
    msg.transforms[0].rotation.x = desired_pose.getRotation().getX();
    msg.transforms[0].rotation.y = desired_pose.getRotation().getY();
    msg.transforms[0].rotation.z = desired_pose.getRotation().getZ();
    msg.transforms[0].rotation.w = desired_pose.getRotation().getW();
    msg.velocities.resize(1);
    msg.velocities[0] = velocity;
    msg.accelerations.resize(1);
    msg.accelerations[0] = acceleration;
    desired_state.publish(msg);
    br.sendTransform(tf::StampedTransform(desired_pose, ros::Time::now(),
                                          "world", "av-desired"));
  }

  void loadMission() {
    std::string mission_file;
    if (!nh_priv.getParam("mission_file", mission_file)) {
      ROS_ERROR("No mission file specified on the parameter server!");
      return;
    }
    
    try {
      mission_config = YAML::LoadFile(mission_file);
      
      cruise_speed = mission_config["speed"].as<double>(5.0);
      initial_hover_time = mission_config["initial_hover_time"].as<double>(5.0);
      waypoint_hover_time = mission_config["waypoint_hover_time"].as<double>(3.0);
      acceleration_limit = mission_config["acceleration_limit"].as<double>(2.0);
      
      ROS_INFO("Mission parameters: speed=%.2f, initial_hover=%.2f, wp_hover=%.2f, accel_limit=%.2f",
               cruise_speed, initial_hover_time, waypoint_hover_time, acceleration_limit);
      
      if (mission_config["initial"]) {
        initial_position =
            tf::Vector3(mission_config["initial"]["x"].as<double>(),
                        mission_config["initial"]["y"].as<double>(),
                        mission_config["initial"]["z"].as<double>());
        ROS_INFO("Loaded initial position: [%.2f, %.2f, %.2f]",
                 initial_position.getX(), initial_position.getY(),
                 initial_position.getZ());
      } else {
        ROS_WARN(
            "No 'initial' key found in mission file; using defaults [0,0,0]");
      }
      
      if (!mission_config["waypoints"]) {
        ROS_ERROR("Mission file does not contain 'waypoints' key");
        return;
      }
      
      for (const auto &wp : mission_config["waypoints"]) {
        Waypoint w;
        w.x = wp["x"].as<double>();
        w.y = wp["y"].as<double>();
        w.z = wp["z"].as<double>();
        waypoints.push_back(w);
      }
      ROS_INFO("Loaded %lu waypoints", waypoints.size());
    } catch (const YAML::Exception &e) {
      ROS_ERROR("Error loading mission file: %s", e.what());
    }
  }

  double missionTime() { return (ros::Time::now() - start_time).toSec(); }

  void currentStateCallback(const nav_msgs::Odometry &msg) {
    current_position =
        tf::Vector3(msg.pose.pose.position.x, msg.pose.pose.position.y,
                    msg.pose.pose.position.z);
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "mission_node");
  ROS_INFO_NAMED("mission_node", "Mission started!");
  missionNode m;
  ros::spin();
  return 0;
}
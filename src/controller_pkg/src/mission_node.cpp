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

  const double initial_hover_time = 5.0;
  const double waypoint_hover_time = 4.0;
  const double waypoint_threshold = 0.5;

  tf::Vector3 initial_position = tf::Vector3(0, 0, 0);
  std::vector<Waypoint> waypoints;

  ros::Time start_time = ros::Time::now();

  tf::Transform desired_pose;
  tf::Vector3 current_position;

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
    
    double cruise_speed = 3; // Max speed in m/s
    double duration; 
    tf::Vector3 start_position = current_position;

    double distance = (target - start_position).length();
    duration = distance / cruise_speed;  // Time to reach target

    tf::Vector3 velocity_vector = (target - start_position).normalize() * cruise_speed;
    
    double start_mission_time = missionTime();
    double progress = 0;
    bool reached_flag = false;
    double hovering_start_time;

    tf::Quaternion q;
    q.setRPY(0, 0, atan2(target.y() - start_position.y(), target.x() - start_position.x()));
    desired_pose.setRotation(q);

    while (ros::ok()) {
        double elapsed_time = missionTime() - start_mission_time;
        progress = elapsed_time / duration;

        if (progress > 1.0) progress = 1.0;
        
        tf::Vector3 new_target = start_position.lerp(target, progress);
        desired_pose.setOrigin(new_target);

        // Adjust velocity smoothly based on distance remaining
        double remaining_distance = (target - new_target).length();
        double speed_factor = remaining_distance / distance; // Slow down as it approaches
        velocity.linear.x = velocity_vector.x() * speed_factor * 0.5;
        velocity.linear.y = velocity_vector.y() * speed_factor * 0.5;
        velocity.linear.z = velocity_vector.z() * speed_factor * 0.5;

        publish();
        ros::spinOnce();
        loop_rate.sleep();

        if (remaining_distance < waypoint_threshold) {
            if (!reached_flag) {
                ROS_INFO("Reached waypoint %d (error=%.2f)", current_wp_index, remaining_distance);
                reached_flag = true;
                hovering_start_time = missionTime();
            }
            if (missionTime() - hovering_start_time >= waypoint_hover_time) {
                break;
            }
        }
    }
}

  void holdLastWaypoint() {
    Waypoint wp = waypoints.back();
    tf::Vector3 target = tf::Vector3(wp.x, wp.y, wp.z);
    desired_pose.setOrigin(target);
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

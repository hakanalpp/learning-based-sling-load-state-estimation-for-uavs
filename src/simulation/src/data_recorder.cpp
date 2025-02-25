#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sys/stat.h> // Required for mkdir

class DataRecorder {
public:
  DataRecorder() : it_(nh_), rate_(10) { // Corrected initialization
    // Subscribe to image topics
    left_image_sub_ = it_.subscribe("/realsense/rgb/left_image_raw", 1,
                                    &DataRecorder::leftImageCallback, this);
    right_image_sub_ = it_.subscribe("/realsense/rgb/right_image_raw", 1,
                                     &DataRecorder::rightImageCallback, this);
  }

  void spin() {
    while (ros::ok()) {
      ros::spinOnce();
      rate_.sleep();
    }
  }

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber left_image_sub_, right_image_sub_;
  ros::Rate rate_; // Loop at 10 Hz
  int left_image_count_ = 0, right_image_count_ = 0;

  void leftImageCallback(const sensor_msgs::ImageConstPtr &msg) {
    saveImage(msg, "left", left_image_count_);
    left_image_count_++;
  }

  void rightImageCallback(const sensor_msgs::ImageConstPtr &msg) {
    saveImage(msg, "right", right_image_count_);
    right_image_count_++;
  }

  void saveImage(const sensor_msgs::ImageConstPtr &msg, const std::string &side,
                 int count) {
    try {
      cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
      std::string directory = "/home/alp/noetic_ws/src/simulation/images/";
      createDirectoryIfNotExists(directory); // Ensure directory exists

      std::string filename =
          directory + side + "_" + std::to_string(count) + ".jpg";

      if (!cv::imwrite(filename, image)) {
        ROS_ERROR_STREAM("Failed to save image: " << filename);
      } else {
        ROS_INFO_STREAM("Saved image: " << filename);
      }
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void createDirectoryIfNotExists(const std::string &directory) {
    struct stat info;
    if (stat(directory.c_str(), &info) != 0) { // Directory does not exist
      ROS_WARN_STREAM("Directory does not exist. Creating: " << directory);
      mkdir(directory.c_str(), 0777); // Create the directory
    }
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "image_recorder");
  DataRecorder recorder;
  recorder.spin();
  return 0;
}

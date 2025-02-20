cmake_minimum_required(VERSION 2.8.3)
project(controller_pkg)

# Force C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed")

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  rospy 
  std_msgs 
  geometry_msgs 
  mav_msgs
  nav_msgs
  eigen_conversions
  tf_conversions
)

catkin_package(CATKIN_DEPENDS 
  roscpp 
  std_msgs 
  mav_msgs 
  tf
  eigen_conversions
  tf_conversions
)

find_package(Eigen3 REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})

add_executable(controller_node src/controller_node.cpp)
add_dependencies(controller_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(controller_node ${catkin_LIBRARIES})

add_executable(traj_publisher src/traj_publisher.cpp)
add_dependencies(traj_publisher ${catkin_EXPORTED_TARGETS})
target_link_libraries(traj_publisher ${catkin_LIBRARIES})


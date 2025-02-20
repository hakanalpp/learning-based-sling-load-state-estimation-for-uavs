catkin_make_isolated --pkg simulation -DCMAKE_BUILD_TYPE=Release &&
source ~/noetic_ws/devel_isolated/setup.bash &&
roslaunch simulation simulation.launch
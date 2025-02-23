catkin_make_isolated --pkg simulation controller_pkg -DCMAKE_BUILD_TYPE=Release &&
source ~/noetic_ws/devel_isolated/setup.bash &&
roslaunch simulation simulation.launch
cmake_minimum_required(VERSION 2.8.3)
project(simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if (NOT COMPILER_SUPPORTS_CXX17)
    message(FATAL_ERROR "Compiler does not support C++17!")
endif()

add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic -Wl,--no-as-needed)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    image_transport
    fla_utils
    fla_msgs
    cv_bridge
    message_generation
    mav_msgs
    nav_msgs
    visualization_msgs
    tf2_geometry_msgs
)

catkin_package(
    CATKIN_DEPENDS mav_msgs
)

include(ExternalProject)
ExternalProject_Add(libsocket
    GIT_REPOSITORY https://github.com/dermesser/libsocket
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CATKIN_DEVEL_PREFIX} -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_SHARED_LIBS=ON
    BUILD_COMMAND make
    INSTALL_COMMAND make install
)

find_package(Eigen3 REQUIRED)

include_directories(
    ${CATKIN_DEVEL_PREFIX}/include
    ${catkin_INCLUDE_DIRS}
    tcpimage
)

link_directories(${CATKIN_DEVEL_PREFIX}/lib)

add_subdirectory(tcpimage)

add_executable(data_recorder src/data_recorder.cpp)
add_executable(unity_ros src/unity_ros.cpp)
add_executable(state_estimate_corruptor_node src/state_estimate_corruptor_node.cpp)
add_executable(w_to_unity src/w_to_unity.cpp)

target_link_libraries(unity_ros ${catkin_LIBRARIES} tcpstreamreader tcpimage)
target_link_libraries(state_estimate_corruptor_node ${catkin_LIBRARIES})
target_link_libraries(w_to_unity ${catkin_LIBRARIES} socket++ -pthread)
target_link_libraries(data_recorder ${catkin_LIBRARIES})

add_dependencies(w_to_unity libsocket ${catkin_EXPORTED_TARGETS})


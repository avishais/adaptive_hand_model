cmake_minimum_required(VERSION 2.8)

project(ExampleProject)

# Find the Shugun libraries and includes
# set Shark_DIR to the proper location of Shark
find_package(shogun REQUIRED)
include(${SHOGUN_USE_FILE})

# Executable
add_executable(ExampleProject test.cpp)
set_property(TARGET ExampleProject PROPERTY CXX_STANDARD 11)
target_link_libraries(ExampleProject ${SHOGUN_LIBRARIES})
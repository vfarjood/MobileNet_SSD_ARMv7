cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 17)
message("CMAKE_CXX_STANDARD = ${CMAKE_CXX_STANDARD}")

project(MobileNet)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find all executables
file(GLOB project_SRCS src/main.cpp)

# Add project executable
add_executable(app ${project_SRCS})
target_link_libraries(app ${OpenCV_LIBRARIES})


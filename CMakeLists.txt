cmake_minimum_required(VERSION 3.15)

project(SYCL_BFS)

set(CMAKE_CXX_STANDARD 17)

# set includes
include_directories(include)

# set include directory
include_directories(include)

# ask for type of GPU
set (SYCL_TARGET "intel_gpu_acm_g10" CACHE STRING "Target for SYCL")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${SYCL_TARGET}")

# add target
add_executable(simple_bfs src/simple_bfs_main.cpp)
add_executable(multi_bfs src/multi_bfs_main.cpp)
cmake_minimum_required(VERSION 3.15)

project(SYCL_BFS)

option(SYCL_BFS_COMPRESSED_GRAPH off "If on, graphs will be compressed on a single vector")
if (SYCL_BFS_COMPRESSED_GRAPH)
    add_compile_definitions(SYCL_BFS_COMPRESSED_GRAPH)
endif()

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
add_executable(bottomup_bfs src/bottom_up_bfs_main.cpp)
add_executable(frontier_bfs src/frontier_bfs_main.cpp)

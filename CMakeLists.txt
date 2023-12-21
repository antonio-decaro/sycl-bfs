cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

project(SYCL_BFS)

option(SYCL_BFS_COMPRESSED_GRAPH off "If on, graphs will be compressed on a single vector")
if (SYCL_BFS_COMPRESSED_GRAPH)
    add_compile_definitions(SYCL_BFS_COMPRESSED_GRAPH)
endif()

option(SUPPORTS_SG_8 on "If on, the device supports Sub-Group size of 8")
if (SUPPORTS_SG_8)
    add_compile_definitions(SUPPORTS_SG_8)
endif()

# set includes
include_directories(include)

# ask for type of GPU
set (SYCL_TARGET "intel_gpu_acm_g10" CACHE STRING "Target for SYCL")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${SYCL_TARGET}")

# add target
add_executable(main src/bottom_up_bfs_main.cpp)

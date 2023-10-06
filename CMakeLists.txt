cmake_minimum_required(VERSION 3.15)

project(SYCL_BFS)

set(CMAKE_CXX_STANDARD 17)

# set includes
include_directories(include)

# set sources
set(SOURCES
    src/main.cpp
    src/bfs.cpp
    src/tile_bfs.cpp
)

# set includes
set(INCLUDES
    include/arg_parse.hpp
    include/bfs.hpp
    include/host_data.hpp
    include/sycl_data.hpp
    include/types.hpp
    include/utils.hpp
    include/kernel_sizes.hpp
)

# ask for type of GPU
set (SYCL_TARGET "intel_gpu_acm_g10" CACHE STRING "Target for SYCL")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${SYCL_TARGET}")

# add target
add_executable(sycl_bfs ${SOURCES} ${INCLUDES})

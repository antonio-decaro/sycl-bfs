cmake_minimum_required(VERSION 3.15)

project(SYCL_BFS)

set(CMAKE_CXX_STANDARD 17)

# set includes
include_directories(include)

# set sources
set(SIMPLE_BFS_SRC
    src/simple_bfs_main.cpp
    src/impl/bfs.cpp
)

set (MULTI_SIMPLE_BFS_SRC
    src/multi_bfs_main.cpp
    src/impl/mul_bfs.cpp
)

set (TILED_BFS_SRC
    src/tiled_bfs_main.cpp
    src/impl/tiled_bfs.cpp
)

# set includes
set(INCLUDES
    include/arg_parse.hpp
    include/bfs.hpp
    include/host_data.hpp
    include/types.hpp
    include/utils.hpp
    include/utils_tiles.hpp
    include/kernel_sizes.hpp
)

# ask for type of GPU
set (SYCL_TARGET "intel_gpu_acm_g10" CACHE STRING "Target for SYCL")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${SYCL_TARGET}")

# add target
add_executable(simple_bfs ${SIMPLE_BFS_SRC} ${INCLUDES})
add_executable(multi_bfs ${MULTI_SIMPLE_BFS_SRC} ${INCLUDES})
add_executable(tiled_bfs ${TILED_BFS_SRC} ${INCLUDES})

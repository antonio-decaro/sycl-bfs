cmake_minimum_required(VERSION 3.15)

project(SYGraph VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ask for type of GPU
set (SYCL_TARGET "intel_gpu_acm_g10" CACHE STRING "Target for SYCL")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${SYCL_TARGET}")

add_library(sygraph INTERFACE)
target_include_directories(sygraph INTERFACE library/include)

option(SUPPORTS_SG_8 on "If on, the device supports Sub-Group size of 8")
if (SUPPORTS_SG_8)
    add_compile_definitions(SUPPORTS_SG_8)
endif()

option(MSM_BUILD_SAMPLES "Build samples" ON)
if(MSM_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
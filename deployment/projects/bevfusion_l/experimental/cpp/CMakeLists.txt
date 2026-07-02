cmake_minimum_required(VERSION 3.18)
project(bevfusion_int8_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# Dependencies
# ---------------------------------------------------------

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()
if(TENSORRT_ROOT)
    include_directories(${TENSORRT_ROOT}/include)
    link_directories(${TENSORRT_ROOT}/lib)
endif()

# libspconv
set(LIBSPCONV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv"
    CACHE PATH "libspconv root (contains include/ and lib/)")

if(NOT EXISTS "${LIBSPCONV_ROOT}/include/spconv/engine.hpp")
    message(FATAL_ERROR
        "libspconv not found at ${LIBSPCONV_ROOT}. "
        "Set -DLIBSPCONV_ROOT=/path/to/libspconv or use the default "
        "Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv path."
    )
endif()

include_directories(${LIBSPCONV_ROOT}/include)

# Auto-detect platform + CUDA version for libspconv.so
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(SPCONV_PLATFORM "aarch64")
else()
    set(SPCONV_PLATFORM "x86_64")
endif()

# Try to detect CUDA version
if(DEFINED CUDA_VERSION_MAJOR AND DEFINED CUDA_VERSION_MINOR)
    set(CUDA_VER "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
else()
    set(CUDA_VER "12.8")
endif()

set(SPCONV_LIB_DIR "${LIBSPCONV_ROOT}/lib/${SPCONV_PLATFORM}_cuda${CUDA_VER}")
if(NOT EXISTS "${SPCONV_LIB_DIR}/libspconv.so")
    # Fallback: try common versions
    foreach(ver "13.0" "12.8" "11.4")
        set(_try "${LIBSPCONV_ROOT}/lib/${SPCONV_PLATFORM}_cuda${ver}")
        if(EXISTS "${_try}/libspconv.so")
            set(SPCONV_LIB_DIR "${_try}")
            break()
        endif()
    endforeach()
endif()

message(STATUS "libspconv library: ${SPCONV_LIB_DIR}/libspconv.so")
link_directories(${SPCONV_LIB_DIR})

# ---------------------------------------------------------
# Bridge library
# ---------------------------------------------------------
add_library(bevfusion_int8_bridge SHARED
    libspconv_trt_bridge.cpp
)

target_link_libraries(bevfusion_int8_bridge
    spconv
    nvinfer
    ${CUDA_LIBRARIES}
    cudart
)

target_include_directories(bevfusion_int8_bridge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBSPCONV_ROOT}/include
)

# ---------------------------------------------------------
# Standalone benchmark executable
# ---------------------------------------------------------
add_executable(benchmark_sparse_int8
    benchmark_sparse_int8.cpp
)

target_link_libraries(benchmark_sparse_int8
    bevfusion_int8_bridge
    spconv
    nvinfer
    ${CUDA_LIBRARIES}
    cudart
)

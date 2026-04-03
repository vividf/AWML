cmake_minimum_required(VERSION 3.18)
project(bevfusion_int8_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This file builds **libspconv_trt_bridge** + benchmark against NVIDIA/Lidar-style **prebuilt**
# libspconv (include/spconv/engine.hpp + lib/<arch>_cuda<x.y>/libspconv.so).
#
# The TensorRT **ImplicitGemmInt8** plugin is a **separate** project:
#   cd int8_plugin && mkdir -p build && cd build && cmake .. && make
# It uses pip/conda **spconv** via find_package(spconv) + CMAKE_PREFIX_PATH — no Lidar tree.
#
# If you do not set LIBSPCONV_ROOT, this directory configures successfully but **skips** the
# bridge targets (no fatal error).

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

# libspconv (Lidar 3DSparseConvolution layout only — optional)
set(LIBSPCONV_ROOT "" CACHE PATH
    "Prebuilt libspconv root (include/spconv/engine.hpp, lib/<arch>_cuda<x.y>/libspconv.so). Empty = skip bridge.")

if(LIBSPCONV_ROOT AND EXISTS "${LIBSPCONV_ROOT}/include/spconv/engine.hpp")
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
else()
    message(STATUS "bevfusion_int8_bridge: skipped (set -DLIBSPCONV_ROOT for Lidar libspconv layout).")
    message(STATUS "  TensorRT ImplicitGemmInt8 plugin: cd ${CMAKE_CURRENT_SOURCE_DIR}/int8_plugin && mkdir -p build && cd build && cmake ..")
endif()

# TensorRT ImplicitGemmInt8 plugin (links pip/conda spconv + cumm via find_package).
#
# Build from **this directory** only:
#   mkdir -p build && cd build && cmake .. [-DCMAKE_PREFIX_PATH=/path/to/spconv/prefix] && make
#
# Do not confuse with ../CMakeLists.txt — that optional project builds libspconv_trt_bridge
# against Lidar_AI_Solution prebuilt libspconv when -DLIBSPCONV_ROOT=... is set.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(implicit_gemm_int8_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Hide all symbols by default so dlopen() in the same process as Python spconv
# does not cause symbol conflicts.  Only TRT plugin registration symbols
# (via REGISTER_TENSORRT_PLUGIN) need to be visible.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ─── CUDA ────────────────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ─── TensorRT ────────────────────────────────────────────────────────────────
# Look in common locations; user can override with -DTRT_ROOT=...
set(TRT_ROOT "/usr" CACHE PATH "TensorRT root")
find_path(TRT_INCLUDE_DIR NvInfer.h
  HINTS ${TRT_ROOT}/include /usr/include/x86_64-linux-gnu)
find_library(NVINFER nvinfer HINTS ${TRT_ROOT}/lib /usr/lib/x86_64-linux-gnu)

if(NOT NVINFER)
  message(FATAL_ERROR "TensorRT nvinfer not found. Set -DTRT_ROOT=...")
endif()

# ─── spconv / cumm ──────────────────────────────────────────────────────────
find_package(cumm REQUIRED)
find_package(spconv REQUIRED)

# ─── Build CUDA quantization kernels ────────────────────────────────────────
set(CUDA_NVCC_FLAGS
  "--expt-relaxed-constexpr"
  "-diag-suppress=1675"
  "--extended-lambda"
  "-gencode" "arch=compute_75,code=sm_75"
  "-gencode" "arch=compute_86,code=sm_86"
  "-gencode" "arch=compute_87,code=sm_87"
  "-gencode" "arch=compute_89,code=sm_89"
)

cuda_add_library(int8_quantize_ops SHARED
  quantize_features.cu
)

target_include_directories(int8_quantize_ops PRIVATE
  ${CUDA_INCLUDE_DIRS}
)

# ─── Build plugin shared library ────────────────────────────────────────────
add_library(implicit_gemm_int8_plugin SHARED
  implicit_gemm_int8_plugin.cpp
  implicit_gemm_int8_plugin_creator.cpp
  plugin_registration.cpp
)

add_definitions("-DTV_CUDA")

target_include_directories(implicit_gemm_int8_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(implicit_gemm_int8_plugin PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(implicit_gemm_int8_plugin PRIVATE
  ${NVINFER}
  CUDA::cudart
  int8_quantize_ops
  spconv::spconv
)

# ─── Install ─────────────────────────────────────────────────────────────────
install(TARGETS implicit_gemm_int8_plugin int8_quantize_ops
  LIBRARY DESTINATION lib
)

message(STATUS "=== ImplicitGemmInt8Plugin Build ===")
message(STATUS "TensorRT: ${NVINFER}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "spconv: found via find_package")

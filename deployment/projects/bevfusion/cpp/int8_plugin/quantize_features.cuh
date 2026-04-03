// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QUANTIZE_FEATURES_CUH_
#define QUANTIZE_FEATURES_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

void launch_quantize_features(
  const __half * input, std::int8_t * output, float scale, std::int64_t total_elements,
  cudaStream_t stream);

void launch_quantize_weights_per_channel(
  const __half * input, std::int8_t * output, const float * w_scales, std::int64_t c_out,
  std::int64_t elements_per_channel, cudaStream_t stream);

void launch_compute_w_scales(
  const float * channel_scale, float * w_scales, float output_scale, float input_scale,
  std::int64_t c_out, cudaStream_t stream);

// ONNX stores baked quantized weights as FP32 (exact integers in [-128,127]) for TensorRT FP16
// engines; cast to int8 in the plugin enqueue.
void launch_cast_float_weights_to_int8(
  const float * input, std::int8_t * output, std::int64_t total_elements, cudaStream_t stream);

#endif  // QUANTIZE_FEATURES_CUH_

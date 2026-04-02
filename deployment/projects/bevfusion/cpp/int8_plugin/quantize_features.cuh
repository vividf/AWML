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

// Fold output dequant into per-channel scale/bias for the s8s8f16 epilogue (alpha unused there).
void launch_fuse_output_scale_into_gemm_scale_bias(
  const float * channel_scale, const float * bias_scaled, float output_scale,
  float * gemm_channel_scale_out, float * gemm_bias_out, std::int64_t c_out, cudaStream_t stream);

#endif  // QUANTIZE_FEATURES_CUH_

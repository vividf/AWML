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

#include "implicit_gemm_int8_plugin.hpp"

#include "quantize_features.cuh"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <spconvlib/cumm/conv/main/ConvMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <algorithm>
#include <atomic>

// Safe assertion for noexcept TRT plugin methods — calls abort() instead of throw.
#ifndef PLUGIN_ASSERT
#define PLUGIN_ASSERT(x)                                                                 \
  do {                                                                                   \
    if (!(x)) {                                                                          \
      std::fprintf(stderr, "[ImplicitGemmInt8Plugin] ASSERT FAILED: %s (%s:%d)\n", #x,  \
        __FILE__, __LINE__);                                                             \
      std::abort();                                                                      \
    }                                                                                    \
  } while (0)
#endif

#ifndef PLUGIN_VALIDATE
#define PLUGIN_VALIDATE(x) PLUGIN_ASSERT(x)
#endif

namespace
{
void caughtError(std::exception const & e)
{
  std::fprintf(stderr, "[ImplicitGemmInt8Plugin] %s\n", e.what());
}

bool int8_gemm_debug_env_enabled() noexcept
{
  char const * v = std::getenv("BEVFUSION_INT8_GEMM_DEBUG");
  if (v == nullptr || v[0] == '\0') {
    return false;
  }
  char c = v[0];
  return c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T';
}

int int8_gemm_debug_max_logs() noexcept
{
  char const * v = std::getenv("BEVFUSION_INT8_GEMM_DEBUG_MAX");
  if (v == nullptr || v[0] == '\0') {
    return 60;
  }
  char * end = nullptr;
  errno = 0;
  long n = std::strtol(v, &end, 10);
  if (errno != 0 || end == v || n <= 0) {
    return 60;
  }
  return static_cast<int>(n);
}

std::atomic<int> g_int8_gemm_debug_seq{0};

// D2H copy + host stats (debug only). Syncs `stream` so TRT enqueue ordering stays defined.
void int8_gemm_debug_dump_fp16_output(
  void const * d_ptr, std::int64_t n_elements, std::string const & layer_name,
  std::int64_t num_act_out, std::int64_t c_out, int seq, float input_scale, float output_scale,
  cudaStream_t stream) noexcept
{
  if (d_ptr == nullptr || n_elements <= 0) {
    return;
  }
  try {
    std::vector<__half> host(static_cast<std::size_t>(n_elements));
    cudaError_t st = cudaMemcpyAsync(
      host.data(), d_ptr, static_cast<std::size_t>(n_elements) * sizeof(__half),
      cudaMemcpyDeviceToHost, stream);
    if (st != cudaSuccess) {
      std::fprintf(
        stderr, "[BEVFUSION_INT8_GEMM_DEBUG] seq=%d layer=%s cudaMemcpyAsync failed: %s\n", seq,
        layer_name.c_str(), cudaGetErrorString(st));
      return;
    }
    st = cudaStreamSynchronize(stream);
    if (st != cudaSuccess) {
      std::fprintf(
        stderr, "[BEVFUSION_INT8_GEMM_DEBUG] seq=%d layer=%s cudaStreamSynchronize failed: %s\n", seq,
        layer_name.c_str(), cudaGetErrorString(st));
      return;
    }

    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    double sum_abs = 0.0;
    std::int64_t nonzero = 0;
    std::int64_t nan_count = 0;
    std::int64_t inf_count = 0;
    for (std::int64_t i = 0; i < n_elements; ++i) {
      float const f = __half2float(host[static_cast<std::size_t>(i)]);
      if (std::isnan(f)) {
        ++nan_count;
        continue;
      }
      if (std::isinf(f)) {
        ++inf_count;
        continue;
      }
      vmin = std::min(vmin, f);
      vmax = std::max(vmax, f);
      sum += static_cast<double>(f);
      sum_abs += static_cast<double>(std::fabs(f));
      if (f != 0.0f) {
        ++nonzero;
      }
    }
    std::int64_t const finite = n_elements - nan_count - inf_count;
    double const mean = finite > 0 ? sum / static_cast<double>(finite) : 0.0;
    double const abs_mean = finite > 0 ? sum_abs / static_cast<double>(finite) : 0.0;

    std::fprintf(
      stderr,
      "[BEVFUSION_INT8_GEMM_DEBUG] seq=%d layer=%s out_shape=[%ld,%ld] n=%ld "
      "input_scale=%.6f output_scale=%.6f min=%.6f max=%.6f mean=%.6f abs_mean=%.6f "
      "nonzero=%ld/%ld nan=%ld inf=%ld\n",
      seq, layer_name.c_str(), static_cast<long>(num_act_out), static_cast<long>(c_out),
      static_cast<long>(n_elements), input_scale, output_scale, vmin, vmax, mean, abs_mean,
      static_cast<long>(nonzero), static_cast<long>(n_elements), static_cast<long>(nan_count),
      static_cast<long>(inf_count));
  } catch (...) {
    std::fprintf(stderr, "[BEVFUSION_INT8_GEMM_DEBUG] seq=%d layer=%s stats: host alloc failed\n", seq,
      layer_name.c_str());
  }
}
}  // namespace

// Alloc keys — spconv headers define SPCONV_ALLOC_FEATURES etc. as macros.
// If not already defined, provide fallbacks.
#ifndef SPCONV_ALLOC_FEATURES
#define SPCONV_ALLOC_FEATURES "features"
#endif
#ifndef SPCONV_ALLOC_FILTERS
#define SPCONV_ALLOC_FILTERS "filters"
#endif
#ifndef SPCONV_ALLOC_OUT_FEATURES
#define SPCONV_ALLOC_OUT_FEATURES "out_features"
#endif

namespace nvinfer1::plugin
{

ImplicitGemmInt8Plugin::ImplicitGemmInt8Plugin(
  const std::string & name, ImplicitGemmInt8Parameters const & params)
: layer_name_{name}, params_{params}
{
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
  using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;

  initFieldsToSerialize();
  arch_ = ConvGemmOps::get_compute_capability();
  tuner_int8_ptr_ = std::make_unique<ConvTunerSimple>(ConvMain::get_all_conv_algo_desp());
}

ImplicitGemmInt8Plugin::~ImplicitGemmInt8Plugin()
{
  releaseConstantCache();
}

void ImplicitGemmInt8Plugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back("act_alpha", &params_.act_alpha, PluginFieldType::kFLOAT32, 1);
  data_to_serialize_.emplace_back("act_beta", &params_.act_beta, PluginFieldType::kFLOAT32, 1);
  data_to_serialize_.emplace_back("is_subm", &params_.is_subm, PluginFieldType::kINT32, 1);
  data_to_serialize_.emplace_back(
    "output_scale", &params_.output_scale, PluginFieldType::kFLOAT32, 1);
  data_to_serialize_.emplace_back(
    "input_scale", &params_.input_scale, PluginFieldType::kFLOAT32, 1);

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

void ImplicitGemmInt8Plugin::releaseConstantCache() noexcept
{
  if (cached_weight_int8_ptr_ != nullptr) {
    cudaFree(cached_weight_int8_ptr_);
    cached_weight_int8_ptr_ = nullptr;
  }
  if (cached_w_scales_ptr_ != nullptr) {
    cudaFree(cached_w_scales_ptr_);
    cached_w_scales_ptr_ = nullptr;
  }
  if (cached_gemm_bias_ptr_ != nullptr) {
    cudaFree(cached_gemm_bias_ptr_);
    cached_gemm_bias_ptr_ = nullptr;
  }
  cache_initialized_ = false;
  cache_mode_logged_ = false;
  cached_c_out_ = 0;
  cached_k1_ = 0;
  cached_k2_ = 0;
  cached_k3_ = 0;
  cached_c_in_ = 0;
  expected_filters_ptr_ = nullptr;
  expected_channel_scale_ptr_ = nullptr;
  expected_bias_scaled_ptr_ = nullptr;
}

std::int32_t ImplicitGemmInt8Plugin::initializeConstantCache(
  PluginTensorDesc const * input_desc, void const * const * inputs, cudaStream_t stream) noexcept
{
  std::int64_t const c_out = input_desc[IN_FILTERS].dims.d[0];
  std::int64_t const k1 = input_desc[IN_FILTERS].dims.d[1];
  std::int64_t const k2 = input_desc[IN_FILTERS].dims.d[2];
  std::int64_t const k3 = input_desc[IN_FILTERS].dims.d[3];
  std::int64_t const c_in = input_desc[IN_FEATURES].dims.d[1];
  std::int64_t const k_vol = k1 * k2 * k3;
  std::int64_t const weight_elements = c_out * k_vol * c_in;
  std::int64_t const weight_bytes = weight_elements * static_cast<std::int64_t>(sizeof(std::int8_t));
  std::int64_t const c_out_float_bytes = c_out * static_cast<std::int64_t>(sizeof(float));

  if (inputs[IN_FILTERS] == nullptr || inputs[IN_CHANNEL_SCALE] == nullptr || inputs[IN_BIAS_SCALED] == nullptr) {
    std::fprintf(
      stderr,
      "[ImplicitGemmInt8Plugin] %s: constant-only mode requires non-null filters/channel_scale/"
      "bias_scaled pointers.\n",
      layer_name_.c_str());
    return -1;
  }
  if (weight_bytes <= 0 || c_out_float_bytes <= 0) {
    std::fprintf(
      stderr, "[ImplicitGemmInt8Plugin] %s: invalid cache shape c_out=%ld k=[%ld,%ld,%ld] c_in=%ld\n",
      layer_name_.c_str(), static_cast<long>(c_out), static_cast<long>(k1), static_cast<long>(k2),
      static_cast<long>(k3), static_cast<long>(c_in));
    return -1;
  }

  releaseConstantCache();

  cudaError_t st = cudaMalloc(reinterpret_cast<void **>(&cached_weight_int8_ptr_), static_cast<std::size_t>(weight_bytes));
  if (st != cudaSuccess) {
    std::fprintf(
      stderr, "[ImplicitGemmInt8Plugin] %s: cudaMalloc(weight_int8 cache) failed: %s\n",
      layer_name_.c_str(), cudaGetErrorString(st));
    releaseConstantCache();
    return -1;
  }
  st = cudaMalloc(reinterpret_cast<void **>(&cached_w_scales_ptr_), static_cast<std::size_t>(c_out_float_bytes));
  if (st != cudaSuccess) {
    std::fprintf(
      stderr, "[ImplicitGemmInt8Plugin] %s: cudaMalloc(w_scales cache) failed: %s\n",
      layer_name_.c_str(), cudaGetErrorString(st));
    releaseConstantCache();
    return -1;
  }
  st = cudaMalloc(reinterpret_cast<void **>(&cached_gemm_bias_ptr_), static_cast<std::size_t>(c_out_float_bytes));
  if (st != cudaSuccess) {
    std::fprintf(
      stderr, "[ImplicitGemmInt8Plugin] %s: cudaMalloc(gemm_bias cache) failed: %s\n",
      layer_name_.c_str(), cudaGetErrorString(st));
    releaseConstantCache();
    return -1;
  }

  launch_compute_w_scales(
    reinterpret_cast<const float *>(inputs[IN_CHANNEL_SCALE]), cached_w_scales_ptr_,
    params_.output_scale, params_.input_scale, c_out, stream);
  launch_quantize_weights_per_channel(
    reinterpret_cast<const __half *>(inputs[IN_FILTERS]), cached_weight_int8_ptr_, cached_w_scales_ptr_,
    c_out, k_vol * c_in, stream);
  launch_fuse_output_scale_into_gemm_scale_bias(
    reinterpret_cast<const float *>(inputs[IN_CHANNEL_SCALE]),
    reinterpret_cast<const float *>(inputs[IN_BIAS_SCALED]), params_.output_scale, cached_w_scales_ptr_,
    cached_gemm_bias_ptr_, c_out, stream);

  st = cudaGetLastError();
  if (st != cudaSuccess) {
    std::fprintf(
      stderr, "[ImplicitGemmInt8Plugin] %s: cache init kernel launch failed: %s\n",
      layer_name_.c_str(), cudaGetErrorString(st));
    releaseConstantCache();
    return -1;
  }

  cache_initialized_ = true;
  cached_c_out_ = c_out;
  cached_k1_ = k1;
  cached_k2_ = k2;
  cached_k3_ = k3;
  cached_c_in_ = c_in;
  expected_filters_ptr_ = inputs[IN_FILTERS];
  expected_channel_scale_ptr_ = inputs[IN_CHANNEL_SCALE];
  expected_bias_scaled_ptr_ = inputs[IN_BIAS_SCALED];
  return 0;
}

IPluginCapability * ImplicitGemmInt8Plugin::getCapabilityInterface(
  PluginCapabilityType type) noexcept
{
  try {
    if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild *>(this);
    if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime *>(this);
    PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV3 * ImplicitGemmInt8Plugin::clone() noexcept
{
  try {
    return new ImplicitGemmInt8Plugin{layer_name_, params_};
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * ImplicitGemmInt8Plugin::getPluginName() const noexcept
{
  return kIMPLICIT_GEMM_INT8_PLUGIN_NAME;
}
char const * ImplicitGemmInt8Plugin::getPluginVersion() const noexcept
{
  return kIMPLICIT_GEMM_INT8_PLUGIN_VERSION;
}
char const * ImplicitGemmInt8Plugin::getPluginNamespace() const noexcept
{
  return kIMPLICIT_GEMM_INT8_PLUGIN_NAMESPACE;
}

std::int32_t ImplicitGemmInt8Plugin::getNbOutputs() const noexcept
{
  return NUM_OUTPUTS;
}

std::int32_t ImplicitGemmInt8Plugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * out, std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == NUM_INPUTS);
  PLUGIN_ASSERT(num_outputs == NUM_OUTPUTS);

  // features: [N, C_in]
  PLUGIN_ASSERT(in[IN_FEATURES].desc.dims.nbDims == 2);
  // filters: [C_out, K1, K2, K3, C_in]
  PLUGIN_ASSERT(in[IN_FILTERS].desc.dims.nbDims == 5);
  // pair_fwd: [K_vol, num_act_out]
  PLUGIN_ASSERT(in[IN_PAIR_FWD].desc.dims.nbDims == 2);
  // pair_mask_fwd: [num_act_out, 1]
  PLUGIN_ASSERT(in[IN_PAIR_MASK_FWD].desc.dims.nbDims == 2);
  // mask_argsort_fwd: [num_act_out]
  PLUGIN_ASSERT(in[IN_MASK_ARGSORT_FWD].desc.dims.nbDims == 1);
  // channel_scale: [C_out]
  PLUGIN_ASSERT(in[IN_CHANNEL_SCALE].desc.dims.nbDims == 1);
  // bias_scaled: [C_out]
  PLUGIN_ASSERT(in[IN_BIAS_SCALED].desc.dims.nbDims == 1);

  PLUGIN_ASSERT(
    in[IN_FILTERS].desc.dims.d[4] == in[IN_FEATURES].desc.dims.d[1]);
  PLUGIN_ASSERT(
    in[IN_CHANNEL_SCALE].desc.dims.d[0] == in[IN_FILTERS].desc.dims.d[0]);
  PLUGIN_ASSERT(
    in[IN_BIAS_SCALED].desc.dims.d[0] == in[IN_FILTERS].desc.dims.d[0]);

  std::fprintf(
    stderr,
    "[ImplicitGemmInt8Plugin] %s: configured in constant-only mode (filters/channel_scale/"
    "bias_scaled must remain constant across enqueues).\n",
    layer_name_.c_str());

  return 0;
}

bool ImplicitGemmInt8Plugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out,
  [[maybe_unused]] std::int32_t num_inputs, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  bool supported = in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;

  switch (pos) {
    // Features: FP16 only (quantized to INT8 inside enqueue)
    case IN_FEATURES:
      supported &= in_out[pos].desc.type == nvinfer1::DataType::kHALF;
      break;
    // Filters: same type as features (FP16)
    case IN_FILTERS:
    case OUT_FEATURES:
      supported &= in_out[pos].desc.type == in_out[IN_FEATURES].desc.type;
      break;
    // Index tensors: INT32
    case IN_PAIR_FWD:
    case IN_PAIR_MASK_FWD:
    case IN_MASK_ARGSORT_FWD:
      supported &= in_out[pos].desc.type == nvinfer1::DataType::kINT32;
      break;
    // Scale/bias: FP32
    case IN_CHANNEL_SCALE:
    case IN_BIAS_SCALED:
      supported &= in_out[pos].desc.type == nvinfer1::DataType::kFLOAT;
      break;
    default:
      supported = false;
      break;
  }
  return supported;
}

std::int32_t ImplicitGemmInt8Plugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs,
  DataType const * input_types, [[maybe_unused]] std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_outputs == NUM_OUTPUTS);
  output_types[0] = input_types[IN_FEATURES];  // FP16
  return 0;
}

std::int32_t ImplicitGemmInt8Plugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs,
  [[maybe_unused]] std::int32_t num_shape_inputs, DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == NUM_INPUTS);
  PLUGIN_ASSERT(num_outputs == NUM_OUTPUTS);

  outputs[0].nbDims = 2;
  // num_act_out from pair_mask_fwd dim 0
  outputs[0].d[0] = inputs[IN_PAIR_MASK_FWD].d[0];
  // C_out from filters dim 0
  outputs[0].d[1] = inputs[IN_FILTERS].d[0];

  return 0;
}

std::int32_t ImplicitGemmInt8Plugin::enqueue(
  PluginTensorDesc const * input_desc,
  [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) noexcept
{
  using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;

  // --- extract dimensions ---
  std::int64_t num_act_in = input_desc[IN_FEATURES].dims.d[0];
  std::int64_t c_in = input_desc[IN_FEATURES].dims.d[1];
  std::int64_t c_out = input_desc[IN_FILTERS].dims.d[0];
  std::int64_t k1 = input_desc[IN_FILTERS].dims.d[1];
  std::int64_t k2 = input_desc[IN_FILTERS].dims.d[2];
  std::int64_t k3 = input_desc[IN_FILTERS].dims.d[3];
  std::int64_t k_vol = k1 * k2 * k3;
  std::int64_t num_act_out = input_desc[IN_PAIR_FWD].dims.d[1];

  // --- workspace layout ---
  auto * ws = reinterpret_cast<std::int8_t *>(workspace);
  std::int8_t * feat_int8_ptr = ws;

  // --- 1. Quantize FP16 features → INT8 ---
  launch_quantize_features(
    reinterpret_cast<const __half *>(inputs[IN_FEATURES]), feat_int8_ptr,
    params_.input_scale, num_act_in * c_in, stream);

  const void * expected_filters_ptr = nullptr;
  const void * expected_channel_scale_ptr = nullptr;
  const void * expected_bias_scaled_ptr = nullptr;
  std::int8_t * cached_weight_int8_ptr = nullptr;
  float * cached_w_scales_ptr = nullptr;
  float * cached_gemm_bias_ptr = nullptr;
  std::int64_t cached_c_out = 0;
  std::int64_t cached_c_in = 0;
  std::int64_t cached_k1 = 0;
  std::int64_t cached_k2 = 0;
  std::int64_t cached_k3 = 0;
  {
    std::lock_guard<std::mutex> lock(cache_init_mutex_);
    if (!cache_initialized_ && initializeConstantCache(input_desc, inputs, stream) != 0) {
      return -1;
    }
    if (!cache_mode_logged_) {
      std::fprintf(
        stderr, "[ImplicitGemmInt8Plugin] %s: constant-only cache mode active (version=%s)\n",
        layer_name_.c_str(), kIMPLICIT_GEMM_INT8_PLUGIN_VERSION);
      cache_mode_logged_ = true;
    }
    expected_filters_ptr = expected_filters_ptr_;
    expected_channel_scale_ptr = expected_channel_scale_ptr_;
    expected_bias_scaled_ptr = expected_bias_scaled_ptr_;
    cached_weight_int8_ptr = cached_weight_int8_ptr_;
    cached_w_scales_ptr = cached_w_scales_ptr_;
    cached_gemm_bias_ptr = cached_gemm_bias_ptr_;
    cached_c_out = cached_c_out_;
    cached_c_in = cached_c_in_;
    cached_k1 = cached_k1_;
    cached_k2 = cached_k2_;
    cached_k3 = cached_k3_;
  }
  // Constant-only contract: these three tensors must be constants bound once per engine.
  if (
    inputs[IN_FILTERS] != expected_filters_ptr ||
    inputs[IN_CHANNEL_SCALE] != expected_channel_scale_ptr ||
    inputs[IN_BIAS_SCALED] != expected_bias_scaled_ptr)
  {
    std::fprintf(
      stderr,
      "[ImplicitGemmInt8Plugin] %s: constant-only mode violation. filters/channel_scale/"
      "bias_scaled pointers changed after cache init.\n",
      layer_name_.c_str());
    return -1;
  }
  if (
    c_out != cached_c_out || c_in != cached_c_in || k1 != cached_k1 || k2 != cached_k2 ||
    k3 != cached_k3)
  {
    std::fprintf(
      stderr,
      "[ImplicitGemmInt8Plugin] %s: constant-only mode violation. filter shape changed after cache "
      "init.\n",
      layer_name_.c_str());
    return -1;
  }

  // --- 2. Build tv::Tensors for implicit_gemm ---
  tv::Tensor features_tv = tv::from_blob(feat_int8_ptr, {num_act_in, c_in}, tv::int8, 0);

  tv::Tensor weights_tv = tv::from_blob(cached_weight_int8_ptr, {c_out, k1, k2, k3, c_in}, tv::int8, 0);

  tv::Tensor pair_fwd = tv::from_blob(
    inputs[IN_PAIR_FWD],
    {input_desc[IN_PAIR_FWD].dims.d[0], input_desc[IN_PAIR_FWD].dims.d[1]}, tv::int32, 0);

  tv::Tensor pair_mask_fwd = tv::from_blob(
    inputs[IN_PAIR_MASK_FWD],
    {1, input_desc[IN_PAIR_MASK_FWD].dims.d[0]}, tv::int32, 0);

  tv::Tensor mask_argsort_fwd = tv::from_blob(
    inputs[IN_MASK_ARGSORT_FWD],
    {1, input_desc[IN_MASK_ARGSORT_FWD].dims.d[0]}, tv::int32, 0);

  // Output: FP16
  tv::Tensor out_features =
    tv::from_blob(outputs[0], {num_act_out, c_out}, tv::float16, 0);

  tv::Tensor mask_tensor = tv::zeros({1}, tv::uint32, -1);
  mask_tensor.data_ptr<uint32_t>()[0] = 0xffffffff;

  tv::Tensor channel_scale_tv = tv::from_blob(cached_w_scales_ptr, {c_out}, tv::float32, 0);
  tv::Tensor bias_scaled_tv = tv::from_blob(cached_gemm_bias_ptr, {c_out}, tv::float32, 0);

  std::vector<tv::Tensor> pair_mask_splits{pair_mask_fwd};
  std::vector<tv::Tensor> mask_argsort_splits{mask_argsort_fwd};

  // StaticAllocator maps alloc keys → pre-allocated tensors
  std::unordered_map<std::string, tv::Tensor> tensor_dict{
    {SPCONV_ALLOC_FEATURES, features_tv},
    {SPCONV_ALLOC_FILTERS, weights_tv},
    {SPCONV_ALLOC_OUT_FEATURES, out_features}};
  StaticAllocator alloc(tensor_dict);

  ConvGemmOps::implicit_gemm(
    alloc, *tuner_int8_ptr_, features_tv, weights_tv, pair_fwd,
    pair_mask_splits, mask_argsort_splits, static_cast<int>(num_act_out),
    mask_tensor, arch_,
    /*is_train=*/false, /*is_subm=*/static_cast<bool>(params_.is_subm),
    reinterpret_cast<std::uintptr_t>(stream), tv::CUDAKernelTimer(false),
    /*auto_fp32_accum=*/true, /*fp32_accum=*/false,
    /*bias=*/bias_scaled_tv,
    /*act_alpha=*/params_.act_alpha,
    /*act_beta=*/params_.act_beta,
    /*act_type=*/tv::gemm::Activation::kNone,
    /*use_tf32=*/false,
    /*output_scale=*/1.0f,
    /*scale=*/channel_scale_tv,
    /*output_add=*/tv::Tensor(),
    /*output_add_scale=*/0.0f,
    /*output_dtype=*/static_cast<int>(tv::float16));

  if (int8_gemm_debug_env_enabled()) {
    int const seq = g_int8_gemm_debug_seq.fetch_add(1, std::memory_order_relaxed);
    if (seq < int8_gemm_debug_max_logs()) {
      std::int64_t const n_el = num_act_out * c_out;
      int8_gemm_debug_dump_fp16_output(
        outputs[0], n_el, layer_name_, num_act_out, c_out, seq, params_.input_scale,
        params_.output_scale, stream);
    }
  }

  return 0;
}

std::int32_t ImplicitGemmInt8Plugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out,
  [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * ImplicitGemmInt8Plugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * ImplicitGemmInt8Plugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t ImplicitGemmInt8Plugin::getWorkspaceSize(
  DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  // INT8 feature scratch only. Weight/scale/bias are cached persistently in constant-only mode.
  auto align = [](std::int64_t x) -> std::int64_t { return (x + 255) & ~255LL; };

  std::int64_t max_n = inputs[IN_FEATURES].max.d[0];
  std::int64_t c_in = inputs[IN_FEATURES].max.d[1];

  std::int64_t feat_bytes = align(max_n * c_in);
  return static_cast<std::size_t>(feat_bytes);
}

}  // namespace nvinfer1::plugin

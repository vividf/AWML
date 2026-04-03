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

#ifndef IMPLICIT_GEMM_INT8_PLUGIN_HPP_
#define IMPLICIT_GEMM_INT8_PLUGIN_HPP_

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_runtime.h>
#include <spconvlib/cumm/conv/main/ConvMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

constexpr char const * const kIMPLICIT_GEMM_INT8_PLUGIN_NAME{"ImplicitGemmInt8"};
// v3: filter input is FP32 ONNX constants holding exact int8 values (avoids TRT INT8 calib / format issues).
constexpr char const * const kIMPLICIT_GEMM_INT8_PLUGIN_VERSION{"3"};
constexpr char const * const kIMPLICIT_GEMM_INT8_PLUGIN_NAMESPACE{""};

namespace nvinfer1::plugin
{

struct ImplicitGemmInt8Parameters
{
  float act_alpha;
  float act_beta;
  std::int64_t is_subm;
  float output_scale;
  float input_scale;   // input_amax / 127.0 for feature quantization
};

// ImplicitGemmInt8Plugin: FP16 activations in/out; INT8 weights; internal INT8 GEMM (cumm).
//
// Inputs (7):
//   0: features         FP16 [N, C_in]  (quantized to INT8 inside enqueue)
//   1: filters          FP32 [C_out, K1, K2, K3, C_in]  (baked int8 values as float; cast in enqueue)
//   2: pair_fwd         INT32 [K_vol, num_act_out]
//   3: pair_mask_fwd    INT32 [num_act_out, 1]
//   4: mask_argsort_fwd INT32 [num_act_out]
//   5: channel_scale    FP32 [C_out]  = (input_scale * w_scales) / output_scale
//   6: bias_scaled      FP32 [C_out]  = bias / output_scale
//
// Output (1):
//   0: out_features     FP16 [num_act_out, C_out]
//
// Per-channel weight quant uses w_scale[c] = channel_scale[c] * output_scale / input_scale
// (same as previous in-plugin path); ONNX transform bakes int8 weights so enqueue skips
// per-inference weight quantization.
class ImplicitGemmInt8Plugin : public IPluginV3,
                               public IPluginV3OneCore,
                               public IPluginV3OneBuild,
                               public IPluginV3OneRuntime
{
public:
  using ConvTunerSimple = spconvlib::spconv::csrc::sparse::convops::spops::ConvTuner;
  ImplicitGemmInt8Plugin(const std::string & name, ImplicitGemmInt8Parameters const & params);

  ~ImplicitGemmInt8Plugin() override = default;

  IPluginCapability * getCapabilityInterface(PluginCapabilityType type) noexcept override;
  IPluginV3 * clone() noexcept override;

  char const * getPluginName() const noexcept override;
  char const * getPluginVersion() const noexcept override;
  char const * getPluginNamespace() const noexcept override;

  std::int32_t getNbOutputs() const noexcept override;

  std::int32_t configurePlugin(
    DynamicPluginTensorDesc const * in, std::int32_t num_inputs,
    DynamicPluginTensorDesc const * out, std::int32_t num_outputs) noexcept override;

  bool supportsFormatCombination(
    std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
    std::int32_t num_outputs) noexcept override;

  std::int32_t getOutputDataTypes(
    DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
    std::int32_t num_inputs) const noexcept override;

  std::int32_t getOutputShapes(
    DimsExprs const * inputs, std::int32_t num_inputs, DimsExprs const * shape_inputs,
    std::int32_t num_shape_inputs, DimsExprs * outputs, std::int32_t num_outputs,
    IExprBuilder & expr_builder) noexcept override;

  std::int32_t enqueue(
    PluginTensorDesc const * input_desc, PluginTensorDesc const * output_desc,
    void const * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) noexcept override;

  std::int32_t onShapeChange(
    PluginTensorDesc const * in, std::int32_t num_inputs, PluginTensorDesc const * out,
    std::int32_t num_outputs) noexcept override;

  IPluginV3 * attachToContext(IPluginResourceContext * context) noexcept override;
  PluginFieldCollection const * getFieldsToSerialize() noexcept override;

  std::size_t getWorkspaceSize(
    DynamicPluginTensorDesc const * inputs, std::int32_t num_inputs,
    DynamicPluginTensorDesc const * outputs, std::int32_t num_outputs) const noexcept override;

private:
  static constexpr std::int32_t IN_FEATURES{0};
  static constexpr std::int32_t IN_FILTERS{1};
  static constexpr std::int32_t IN_PAIR_FWD{2};
  static constexpr std::int32_t IN_PAIR_MASK_FWD{3};
  static constexpr std::int32_t IN_MASK_ARGSORT_FWD{4};
  static constexpr std::int32_t IN_CHANNEL_SCALE{5};
  static constexpr std::int32_t IN_BIAS_SCALED{6};
  static constexpr std::int32_t OUT_FEATURES{7};

  static constexpr std::int32_t NUM_INPUTS{7};
  static constexpr std::int32_t NUM_OUTPUTS{1};

  void initFieldsToSerialize();

  std::string layer_name_;
  ImplicitGemmInt8Parameters params_;
  std::tuple<int, int> arch_;
  std::vector<nvinfer1::PluginField> data_to_serialize_;
  nvinfer1::PluginFieldCollection fc_to_serialize_;

  std::unique_ptr<ConvTunerSimple> tuner_int8_ptr_{};
};

}  // namespace nvinfer1::plugin

#endif  // IMPLICIT_GEMM_INT8_PLUGIN_HPP_

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

#ifndef IMPLICIT_GEMM_INT8_PLUGIN_CREATOR_HPP_
#define IMPLICIT_GEMM_INT8_PLUGIN_CREATOR_HPP_

#include "implicit_gemm_int8_plugin.hpp"

#include <NvInferRuntime.h>

#include <vector>

namespace nvinfer1::plugin
{

class ImplicitGemmInt8PluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
  ImplicitGemmInt8PluginCreator();
  ~ImplicitGemmInt8PluginCreator() override = default;

  char const * getPluginNamespace() const noexcept override
  {
    return kIMPLICIT_GEMM_INT8_PLUGIN_NAMESPACE;
  }
  char const * getPluginName() const noexcept override
  {
    return kIMPLICIT_GEMM_INT8_PLUGIN_NAME;
  }
  char const * getPluginVersion() const noexcept override
  {
    return kIMPLICIT_GEMM_INT8_PLUGIN_VERSION;
  }

  nvinfer1::PluginFieldCollection const * getFieldNames() noexcept override;

  IPluginV3 * createPlugin(
    char const * name, PluginFieldCollection const * fc, TensorRTPhase phase) noexcept override;

private:
  nvinfer1::PluginFieldCollection fc_;
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};

}  // namespace nvinfer1::plugin

#endif  // IMPLICIT_GEMM_INT8_PLUGIN_CREATOR_HPP_

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

#include "implicit_gemm_int8_plugin_creator.hpp"

#include "implicit_gemm_int8_plugin.hpp"

#include <NvInferRuntimePlugin.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace nvinfer1::plugin
{

REGISTER_TENSORRT_PLUGIN(ImplicitGemmInt8PluginCreator);

ImplicitGemmInt8PluginCreator::ImplicitGemmInt8PluginCreator()
{
  plugin_attributes_.clear();
  plugin_attributes_.emplace_back("act_alpha", nullptr, PluginFieldType::kFLOAT32, 1);
  plugin_attributes_.emplace_back("act_beta", nullptr, PluginFieldType::kFLOAT32, 1);
  plugin_attributes_.emplace_back("is_subm", nullptr, PluginFieldType::kINT32, 1);
  plugin_attributes_.emplace_back("output_scale", nullptr, PluginFieldType::kFLOAT32, 1);
  plugin_attributes_.emplace_back("input_scale", nullptr, PluginFieldType::kFLOAT32, 1);
  plugin_attributes_.emplace_back("timing_enabled", nullptr, PluginFieldType::kINT32, 1);
  plugin_attributes_.emplace_back("timing_max_logs", nullptr, PluginFieldType::kINT32, 1);
  plugin_attributes_.emplace_back("act_type", nullptr, PluginFieldType::kINT32, 1);

  fc_.nbFields = plugin_attributes_.size();
  fc_.fields = plugin_attributes_.data();
}

nvinfer1::PluginFieldCollection const * ImplicitGemmInt8PluginCreator::getFieldNames() noexcept
{
  return &fc_;
}

IPluginV3 * ImplicitGemmInt8PluginCreator::createPlugin(
  char const * name, PluginFieldCollection const * fc, TensorRTPhase phase) noexcept
{
  if (phase == TensorRTPhase::kBUILD || phase == TensorRTPhase::kRUNTIME) {
    try {
      nvinfer1::PluginField const * fields{fc->fields};
      std::int32_t num_fields{fc->nbFields};

      ImplicitGemmInt8Parameters params{};
      params.act_alpha = 0.0f;
      params.act_beta = 0.0f;
      params.is_subm = 0;
      params.output_scale = 1.0f;
      params.input_scale = 1.0f;
      params.timing_enabled = 0;
      params.timing_max_logs = 1000;
      params.act_type = 0;

      for (std::int32_t i = 0; i < num_fields; ++i) {
        const std::string attr_name = fields[i].name;

        if (attr_name == "act_alpha") {
          params.act_alpha = static_cast<float const *>(fields[i].data)[0];
        } else if (attr_name == "act_beta") {
          params.act_beta = static_cast<float const *>(fields[i].data)[0];
        } else if (attr_name == "is_subm") {
          params.is_subm = static_cast<std::int32_t const *>(fields[i].data)[0];
        } else if (attr_name == "output_scale") {
          params.output_scale = static_cast<float const *>(fields[i].data)[0];
        } else if (attr_name == "input_scale") {
          params.input_scale = static_cast<float const *>(fields[i].data)[0];
        } else if (attr_name == "timing_enabled") {
          params.timing_enabled = static_cast<std::int32_t const *>(fields[i].data)[0];
        } else if (attr_name == "timing_max_logs") {
          params.timing_max_logs = static_cast<std::int32_t const *>(fields[i].data)[0];
        } else if (attr_name == "act_type") {
          params.act_type = static_cast<std::int32_t const *>(fields[i].data)[0];
        }
      }

      if (params.timing_max_logs <= 0) {
        params.timing_max_logs = 1000;
      }

      std::fprintf(
        stderr,
        "[ImplicitGemmInt8] %s: is_subm=%ld output_scale=%.6f input_scale=%.6f timing_enabled=%d "
        "timing_max_logs=%d\n",
        name, params.is_subm, params.output_scale, params.input_scale, params.timing_enabled,
        params.timing_max_logs);

      return new ImplicitGemmInt8Plugin{std::string(name), params};
    } catch (std::exception const & e) {
      std::fprintf(stderr, "[ImplicitGemmInt8PluginCreator] %s\n", e.what());
    }
  }
  return nullptr;
}

}  // namespace nvinfer1::plugin

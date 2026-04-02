/*
 * BEVFusion INT8 inference bridge implementation.
 *
 * Sparse encoder: libspconv (INT8 cumm kernels)
 * Dense backbone/neck/head: TensorRT (FP16)
 */

#include "libspconv_trt_bridge.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>

namespace bevfusion {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

static TrtLogger g_trt_logger;

LibspconvTrtBridge::~LibspconvTrtBridge() {
    if (bev_buffer_) cudaFree(bev_buffer_);
    if (trt_output_buffer_) cudaFree(trt_output_buffer_);
    if (trt_context_) trt_context_->destroy();
    if (trt_engine_) trt_engine_->destroy();
    if (trt_runtime_) trt_runtime_->destroy();
}

bool LibspconvTrtBridge::init(const BridgeConfig& config) {
    config_ = config;

    if (!init_sparse_engine(config)) {
        std::cerr << "[bridge] Failed to initialize libspconv sparse engine" << std::endl;
        return false;
    }

    if (!config.dense_trt_engine_path.empty()) {
        if (!init_dense_engine(config)) {
            std::cerr << "[bridge] Failed to initialize TensorRT dense engine" << std::endl;
            return false;
        }
    }

    size_t bev_bytes = sizeof(__half) * config.bev_channels * config.bev_height * config.bev_width;
    cudaMalloc(&bev_buffer_, bev_bytes);

    std::cout << "[bridge] Initialized: sparse=" << config.sparse_onnx_path
              << " precision=" << spconv::get_precision_string(config.sparse_precision)
              << std::endl;

    return true;
}

bool LibspconvTrtBridge::init_sparse_engine(const BridgeConfig& config) {
    sparse_engine_ = spconv::load_engine_from_onnx(
        config.sparse_onnx_path,
        config.sparse_precision
    );
    return sparse_engine_ != nullptr;
}

bool LibspconvTrtBridge::init_dense_engine(const BridgeConfig& config) {
    trt_runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!trt_runtime_) return false;

    std::ifstream fin(config.dense_trt_engine_path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[bridge] Cannot open TRT engine: " << config.dense_trt_engine_path << std::endl;
        return false;
    }

    fin.seekg(0, std::ios::end);
    size_t size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    fin.read(engine_data.data(), size);

    trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!trt_engine_) return false;

    trt_context_ = trt_engine_->createExecutionContext();
    if (!trt_context_) return false;

    int nb = trt_engine_->getNbBindings();
    trt_bindings_.resize(nb, nullptr);

    for (int i = 0; i < nb; ++i) {
        auto dims = trt_engine_->getBindingDimensions(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            vol *= dims.d[d];
        }
        size_t bytes = vol * sizeof(__half);

        if (trt_engine_->bindingIsInput(i)) {
            trt_bindings_[i] = bev_buffer_;
        } else {
            cudaMalloc(&trt_bindings_[i], bytes);
            trt_output_buffer_ = trt_bindings_[i];
        }
    }

    return true;
}

const void* LibspconvTrtBridge::forward(
    const void* features,
    const void* indices,
    int num_voxels,
    void* stream
) {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    auto t0 = std::chrono::high_resolution_clock::now();

    // --- Sparse Encoder (libspconv INT8) ---
    auto* input = sparse_engine_->input(0);
    input->features().reference(
        const_cast<void*>(features),
        {static_cast<int64_t>(num_voxels), static_cast<int64_t>(config_.voxel_feature_dim)},
        spconv::DataType::Float16
    );
    input->indices().reference(
        const_cast<void*>(indices),
        {static_cast<int64_t>(num_voxels), 4},
        spconv::DataType::Int32
    );
    input->set_grid_size(config_.grid_size);

    sparse_engine_->forward(stream);

    auto t1 = std::chrono::high_resolution_clock::now();

    // BEV output from sparse encoder (FP16)
    auto* output = sparse_engine_->output(0);
    const void* bev_features = output->features().ptr<__half>();
    auto bev_out_shape = output->features().shape;

    if (config_.enable_profiling) {
        cudaStreamSynchronize(cu_stream);
        auto t1_sync = std::chrono::high_resolution_clock::now();
        last_timing_.sparse_encoder_ms =
            std::chrono::duration<float, std::milli>(t1_sync - t0).count();
    }

    // --- Dense Engine (TensorRT FP16) ---
    if (trt_context_) {
        size_t bev_bytes = sizeof(__half);
        for (auto d : bev_out_shape) bev_bytes *= d;
        cudaMemcpyAsync(
            bev_buffer_, bev_features, bev_bytes,
            cudaMemcpyDeviceToDevice, cu_stream
        );

        trt_context_->enqueueV2(trt_bindings_.data(), cu_stream, nullptr);

        if (config_.enable_profiling) {
            cudaStreamSynchronize(cu_stream);
            auto t2 = std::chrono::high_resolution_clock::now();
            last_timing_.dense_backbone_ms =
                std::chrono::duration<float, std::milli>(t2 - t1).count();
            last_timing_.total_ms =
                std::chrono::duration<float, std::milli>(t2 - t0).count();
        }

        return trt_output_buffer_;
    }

    // If no dense engine, return BEV features directly
    return bev_features;
}

std::vector<int64_t> LibspconvTrtBridge::bev_shape() const {
    if (sparse_engine_ && sparse_engine_->num_output() > 0) {
        return sparse_engine_->output(0)->features().shape;
    }
    return {1, config_.bev_channels, config_.bev_height, config_.bev_width};
}

}  // namespace bevfusion

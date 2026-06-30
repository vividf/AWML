/*
 * BEVFusion INT8 inference bridge: libspconv sparse encoder + TensorRT dense engine.
 *
 * The sparse encoder runs on NVIDIA's libspconv with INT8 cumm kernels.
 * The dense backbone/neck/head runs on TensorRT (FP16).
 * They connect at the BEV feature map boundary.
 *
 * Build: see CMakeLists.txt in this directory.
 */

#ifndef BEVFUSION_LIBSPCONV_TRT_BRIDGE_HPP
#define BEVFUSION_LIBSPCONV_TRT_BRIDGE_HPP

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <spconv/engine.hpp>

namespace bevfusion {

struct BridgeConfig {
    std::string sparse_onnx_path;
    std::string dense_trt_engine_path;

    spconv::Precision sparse_precision = spconv::Precision::Int8;

    std::vector<int> grid_size = {1440, 1440, 41};

    int max_voxels = 160000;
    int voxel_feature_dim = 5;
    int bev_channels = 256;
    int bev_height = 180;
    int bev_width = 180;

    bool enable_profiling = false;
};

struct InferenceResult {
    float sparse_encoder_ms = 0.0f;
    float dense_backbone_ms = 0.0f;
    float total_ms = 0.0f;
};

class LibspconvTrtBridge {
public:
    LibspconvTrtBridge() = default;
    ~LibspconvTrtBridge();

    LibspconvTrtBridge(const LibspconvTrtBridge&) = delete;
    LibspconvTrtBridge& operator=(const LibspconvTrtBridge&) = delete;

    bool init(const BridgeConfig& config);

    /**
     * Run full BEVFusion inference.
     *
     * @param features  Device pointer to voxel features (FP16), shape [N, C].
     * @param indices   Device pointer to voxel indices (INT32), shape [N, 4].
     * @param num_voxels Number of valid voxels.
     * @param stream    CUDA stream.
     * @return          Device pointer to detection output (from TRT engine).
     */
    const void* forward(
        const void* features,
        const void* indices,
        int num_voxels,
        void* stream = nullptr
    );

    InferenceResult last_timing() const { return last_timing_; }

    std::vector<int64_t> bev_shape() const;

private:
    bool init_sparse_engine(const BridgeConfig& config);
    bool init_dense_engine(const BridgeConfig& config);

    BridgeConfig config_;
    std::shared_ptr<spconv::Engine> sparse_engine_;

    nvinfer1::IRuntime* trt_runtime_ = nullptr;
    nvinfer1::ICudaEngine* trt_engine_ = nullptr;
    nvinfer1::IExecutionContext* trt_context_ = nullptr;

    void* bev_buffer_ = nullptr;
    void* trt_output_buffer_ = nullptr;
    std::vector<void*> trt_bindings_;

    InferenceResult last_timing_;
};

}  // namespace bevfusion

#endif  // BEVFUSION_LIBSPCONV_TRT_BRIDGE_HPP

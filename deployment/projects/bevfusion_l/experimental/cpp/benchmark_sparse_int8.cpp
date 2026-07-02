/*
 * Benchmark: libspconv INT8 sparse encoder vs timing comparison.
 *
 * Usage:
 *   ./benchmark_sparse_int8 \
 *       --sparse-onnx /path/to/sparse_encoder_int8.onnx \
 *       [--dense-trt /path/to/dense_engine.trt] \
 *       [--warmup 10] [--iterations 100] [--num-voxels 40000]
 */

#include "libspconv_trt_bridge.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct Args {
    std::string sparse_onnx;
    std::string dense_trt;
    int warmup = 10;
    int iterations = 100;
    int num_voxels = 40000;
    int voxel_dim = 5;
    bool int8 = true;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sparse-onnx" && i + 1 < argc) args.sparse_onnx = argv[++i];
        else if (a == "--dense-trt" && i + 1 < argc) args.dense_trt = argv[++i];
        else if (a == "--warmup" && i + 1 < argc) args.warmup = std::stoi(argv[++i]);
        else if (a == "--iterations" && i + 1 < argc) args.iterations = std::stoi(argv[++i]);
        else if (a == "--num-voxels" && i + 1 < argc) args.num_voxels = std::stoi(argv[++i]);
        else if (a == "--fp16") args.int8 = false;
        else if (a == "--help") {
            std::cout << "Usage: benchmark_sparse_int8 --sparse-onnx <path> [options]\n"
                      << "  --sparse-onnx <path>   Custom libspconv ONNX (required)\n"
                      << "  --dense-trt <path>     TRT engine for dense backbone (optional)\n"
                      << "  --warmup <N>           Warmup iterations (default: 10)\n"
                      << "  --iterations <N>       Benchmark iterations (default: 100)\n"
                      << "  --num-voxels <N>       Simulated voxel count (default: 40000)\n"
                      << "  --fp16                 Use FP16 precision instead of INT8\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.sparse_onnx.empty()) {
        std::cerr << "Error: --sparse-onnx is required. Use --help for usage.\n";
        return 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate dummy input data
    size_t feat_bytes = args.num_voxels * args.voxel_dim * sizeof(__half);
    size_t idx_bytes = args.num_voxels * 4 * sizeof(int32_t);

    void* d_features = nullptr;
    void* d_indices = nullptr;
    cudaMalloc(&d_features, feat_bytes);
    cudaMalloc(&d_indices, idx_bytes);
    cudaMemset(d_features, 0, feat_bytes);
    cudaMemset(d_indices, 0, idx_bytes);

    // Initialize bridge
    bevfusion::BridgeConfig config;
    config.sparse_onnx_path = args.sparse_onnx;
    config.dense_trt_engine_path = args.dense_trt;
    config.sparse_precision = args.int8 ? spconv::Precision::Int8 : spconv::Precision::Float16;
    config.voxel_feature_dim = args.voxel_dim;
    config.enable_profiling = true;

    bevfusion::LibspconvTrtBridge bridge;
    if (!bridge.init(config)) {
        std::cerr << "Failed to initialize bridge\n";
        return 1;
    }

    auto bev_shape = bridge.bev_shape();
    std::cout << "BEV output shape: [";
    for (size_t i = 0; i < bev_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << bev_shape[i];
    }
    std::cout << "]\n";

    std::string prec_str = args.int8 ? "INT8" : "FP16";
    std::cout << "\nBenchmark: " << prec_str << " sparse encoder, "
              << args.num_voxels << " voxels\n";

    // Warmup
    std::cout << "Warmup: " << args.warmup << " iterations\n";
    for (int i = 0; i < args.warmup; ++i) {
        bridge.forward(d_features, d_indices, args.num_voxels, stream);
        cudaStreamSynchronize(stream);
    }

    // Benchmark
    std::vector<float> sparse_times;
    std::vector<float> total_times;

    for (int i = 0; i < args.iterations; ++i) {
        bridge.forward(d_features, d_indices, args.num_voxels, stream);
        cudaStreamSynchronize(stream);

        auto timing = bridge.last_timing();
        sparse_times.push_back(timing.sparse_encoder_ms);
        total_times.push_back(timing.total_ms > 0 ? timing.total_ms : timing.sparse_encoder_ms);
    }

    // Statistics
    auto stats = [](const std::vector<float>& v) {
        float sum = std::accumulate(v.begin(), v.end(), 0.0f);
        float mean = sum / v.size();
        float sq_sum = 0;
        for (auto x : v) sq_sum += (x - mean) * (x - mean);
        float std_dev = std::sqrt(sq_sum / v.size());
        float min_val = *std::min_element(v.begin(), v.end());
        float max_val = *std::max_element(v.begin(), v.end());
        return std::make_tuple(mean, std_dev, min_val, max_val);
    };

    auto [sp_mean, sp_std, sp_min, sp_max] = stats(sparse_times);
    auto [tot_mean, tot_std, tot_min, tot_max] = stats(total_times);

    std::cout << "\n=== Results (" << args.iterations << " iterations) ===\n";
    std::cout << "Sparse Encoder (" << prec_str << "):\n";
    std::cout << "  Mean: " << sp_mean << " ms\n";
    std::cout << "  Std:  " << sp_std << " ms\n";
    std::cout << "  Min:  " << sp_min << " ms\n";
    std::cout << "  Max:  " << sp_max << " ms\n";

    if (!args.dense_trt.empty()) {
        std::cout << "Total (sparse + dense):\n";
        std::cout << "  Mean: " << tot_mean << " ms\n";
        std::cout << "  Std:  " << tot_std << " ms\n";
    }

    // Cleanup
    cudaFree(d_features);
    cudaFree(d_indices);
    cudaStreamDestroy(stream);

    return 0;
}

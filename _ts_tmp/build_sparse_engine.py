"""Throwaway: build a TRT engine from a (possibly trainStation-stripped) sparse ONNX.

Usage: python build_sparse_engine.py <in.onnx> <out.engine>
Auto-derives an optimization profile: dynamic dims -> (1, 32000, 256000),
except the voxel row dim -> (1, 64000, 256000). Verbose builder log to stderr.
"""

import ctypes
import sys

import tensorrt as trt

IN_ONNX, OUT_ENGINE = sys.argv[1], sys.argv[2]
PLUGIN_SO = "/opt/plugins/libautoware_tensorrt_plugins.so"

ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)


class StderrLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity <= trt.ILogger.Severity.INFO:
            print(f"[TRT {severity}] {msg}", file=sys.stderr)


logger = StderrLogger()
trt.init_libnvinfer_plugins(logger, "")

builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
with open(IN_ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("PARSE ERROR:", parser.get_error(i), file=sys.stderr)
        sys.exit(1)

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

profile = builder.create_optimization_profile()
print(f"=== {network.num_inputs} network inputs ===", file=sys.stderr)
for i in range(network.num_inputs):
    inp = network.get_input(i)
    shp = list(inp.shape)
    mn, op, mx = [], [], []
    for d in shp:
        if d == -1:
            # Uniform bounds for every dynamic dim. voxels/coors/num_points_per_voxel
            # share the SAME onnx dim_param ('voxels_num'); TRT requires their profiles
            # to match exactly, so all dynamic dims must use identical bounds here.
            mn.append(1)
            op.append(64000)
            mx.append(256000)
        else:
            mn.append(d)
            op.append(d)
            mx.append(d)
    profile.set_shape(inp.name, tuple(mn), tuple(op), tuple(mx))
    print(f"  [{i}] {inp.name} shape={shp} min{mn} opt{op} max{mx}", file=sys.stderr)
config.add_optimization_profile(profile)

print("=== building ===", file=sys.stderr)
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print("BUILD FAILED", file=sys.stderr)
    sys.exit(2)
with open(OUT_ENGINE, "wb") as f:
    f.write(bytes(serialized))
print(f"BUILD OK -> {OUT_ENGINE} bytes={len(bytes(serialized))}", file=sys.stderr)

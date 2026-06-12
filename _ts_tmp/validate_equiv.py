"""Correctness check: baseline sparse engine vs trainStation-stripped engine.

Feeds BOTH engines the SAME synthetic sparse input. The modified engine additionally
gets the 4 down-sample rulebooks, precomputed here with the same spconv routine the
plugin uses (sparse_functional.GetIndicePairsImplicitGemm). If the two ``lidar_bev``
outputs match, the precompute layout + the graph surgery are correct.

Run inside the awml-bevfusion container from /workspace.
"""

import ctypes
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/projects/SparseConvolution")

import tensorrt as trt
from spconv.core import ConvAlgo
from spconv.tools import CUDAKernelTimer

from projects.SparseConvolution.sparse_functional import GetIndicePairsImplicitGemm


def graph_input_zyx_to_model_indices_xyz(coors):
    """[M,3] graph input [z,y,x] -> model indices [x,y,z] (flip last dim)."""
    if coors.ndim != 2 or coors.shape[1] != 3:
        return coors
    return coors.flip(dims=[-1]).contiguous()


DEV = "cuda:0"
BASE_ENG = "_ts_tmp/sparse_base.engine"
MOD_ENG = "_ts_tmp/sparse_nots.engine"
PLUGIN = "/opt/plugins/libautoware_tensorrt_plugins.so"

ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_GLOBAL)


class L(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, sev, msg):
        if sev <= trt.ILogger.Severity.ERROR:
            print(f"[TRT {sev}] {msg}", file=sys.stderr)


LOGGER = L()
trt.init_libnvinfer_plugins(LOGGER, "")

# ---- down-sample layer attributes (from the ONNX nodes) ----
DOWNS = [
    dict(
        tag="encoder_layer1",
        ksize=[3, 3, 3],
        stride=[2, 2, 2],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        spatial=[1440, 1440, 41],
    ),
    dict(
        tag="encoder_layer2",
        ksize=[3, 3, 3],
        stride=[2, 2, 2],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        spatial=[720, 720, 21],
    ),
    dict(
        tag="encoder_layer3",
        ksize=[3, 3, 3],
        stride=[2, 2, 2],
        padding=[1, 1, 0],
        dilation=[1, 1, 1],
        spatial=[360, 360, 11],
    ),
    dict(
        tag="conv_out", ksize=[1, 1, 3], stride=[1, 1, 2], padding=[0, 0, 0], dilation=[1, 1, 1], spatial=[180, 180, 5]
    ),
]


def make_synthetic(n=40000, seed=0):
    g = torch.Generator().manual_seed(seed)
    # unique [z,y,x] coords within spatial bounds (graph-input order)
    z = torch.randint(0, 41, (n * 2, 1), generator=g)
    y = torch.randint(0, 1440, (n * 2, 1), generator=g)
    x = torch.randint(0, 1440, (n * 2, 1), generator=g)
    zyx = torch.cat([z, y, x], dim=1)
    zyx = torch.unique(zyx, dim=0)[:n].contiguous().to(torch.int32)
    n = zyx.shape[0]
    voxels = torch.rand(n, 10, 5, generator=g).to(torch.float32) * 5.0
    npp = torch.randint(1, 11, (n,), generator=g).to(torch.int32)
    return voxels, zyx, npp


def precompute_rulebooks(coors_zyx):
    """coors_zyx: [N,3] int32 graph-input order. Returns dict (tag,oi)->cuda tensor."""
    coors = coors_zyx.to(DEV)
    coords_xyz = graph_input_zyx_to_model_indices_xyz(coors)  # [N,3] (x,y,z)
    batch = torch.zeros(coords_xyz.shape[0], 1, dtype=torch.int32, device=DEV)
    cur = torch.cat([batch, coords_xyz], dim=1).contiguous().to(torch.int32)  # [N,4] (b,x,y,z)

    out = {}
    timer = CUDAKernelTimer(False)
    for d in DOWNS:
        res = GetIndicePairsImplicitGemm.apply(
            cur,
            1,
            d["spatial"],
            ConvAlgo(1),
            d["ksize"],
            d["stride"],
            d["padding"],
            d["dilation"],
            [0, 0, 0],
            False,
            False,
            False,
            None,
            timer,
        )
        out_inds, pair_fwd, pair_mask, mask_argsort, num_act = res
        n = int(out_inds.shape[0])
        out[(d["tag"], 0)] = out_inds.to(torch.int32).contiguous()
        out[(d["tag"], 1)] = pair_fwd.to(torch.int32).contiguous()
        out[(d["tag"], 2)] = pair_mask.reshape(n, 1).to(torch.int32).contiguous()
        out[(d["tag"], 3)] = mask_argsort.reshape(n).to(torch.int32).contiguous()
        print(
            f"  precompute {d['tag']}: out_inds={tuple(out_inds.shape)} pair_fwd={tuple(pair_fwd.shape)} "
            f"num_act={int(num_act.item())}"
        )
        cur = out_inds.to(torch.int32).contiguous()
    return out


_TRT2TORCH = {trt.int32: torch.int32, trt.float32: torch.float32, trt.float16: torch.float16}


def run_engine(engine_path, feed):
    """feed: dict name->cuda tensor. Returns dict of output name->cpu tensor."""
    rt = trt.Runtime(LOGGER)
    with open(engine_path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    keep = []
    outputs = {}
    with torch.cuda.stream(stream):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            dt = _TRT2TORCH[engine.get_tensor_dtype(name)]
            if mode == trt.TensorIOMode.INPUT:
                t = feed[name].to(DEV).to(dt).contiguous()
                ctx.set_input_shape(name, tuple(t.shape))
                ctx.set_tensor_address(name, t.data_ptr())
                keep.append(t)
            else:
                outputs[name] = None
        # outputs: allocate after all input shapes set
        for name in list(outputs.keys()):
            shp = tuple(ctx.get_tensor_shape(name))
            dt = _TRT2TORCH[engine.get_tensor_dtype(name)]
            o = torch.empty(shp, dtype=dt, device=DEV)
            ctx.set_tensor_address(name, o.data_ptr())
            outputs[name] = o
            keep.append(o)
        ok = ctx.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"execute failed for {engine_path}")
    stream.synchronize()
    return {k: v.float().cpu() for k, v in outputs.items()}


def main():
    voxels, coors, npp = make_synthetic()
    print(f"synthetic: voxels={tuple(voxels.shape)} coors={tuple(coors.shape)} npp={tuple(npp.shape)}")

    base_feed = {"voxels": voxels, "coors": coors, "num_points_per_voxel": npp}
    print("=== baseline engine ===")
    base_out = run_engine(BASE_ENG, base_feed)

    print("=== precompute rulebooks ===")
    rb = precompute_rulebooks(coors)

    # map rulebook tensors to the modified engine input names
    mod_feed = dict(base_feed)
    rt = trt.Runtime(LOGGER)
    with open(MOD_ENG, "rb") as f:
        mod_engine = rt.deserialize_cuda_engine(f.read())
    for i in range(mod_engine.num_io_tensors):
        name = mod_engine.get_tensor_name(i)
        if mod_engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if name in base_feed:
            continue
        # name like .../<tag>/.../GetIndicePairsImplicitGemm_output_<oi>
        oi = int(name.rsplit("_output_", 1)[1])
        tag = next(
            d["tag"] for d in DOWNS if f"/{d['tag']}/" in name or (d["tag"] == "conv_out" and "/conv_out/" in name)
        )
        mod_feed[name] = rb[(tag, oi)]

    print("=== modified engine ===")
    mod_out = run_engine(MOD_ENG, mod_feed)

    print("=== compare lidar_bev ===")
    a = base_out["lidar_bev"]
    b = mod_out["lidar_bev"]
    print(f"baseline {tuple(a.shape)}  modified {tuple(b.shape)}")
    if a.shape != b.shape:
        print("SHAPE MISMATCH")
        return
    diff = (a - b).abs()
    print(f"max abs diff = {diff.max().item():.6f}")
    print(f"mean abs diff = {diff.mean().item():.6f}")
    print(f"baseline range [{a.min().item():.3f},{a.max().item():.3f}]  nonzero={int((a!=0).sum())}")
    rel = diff.max().item() / (a.abs().max().item() + 1e-6)
    print(f"relative max diff = {rel:.6f}")
    print("RESULT:", "MATCH" if diff.max().item() < 1e-2 else "MISMATCH")


if __name__ == "__main__":
    with torch.no_grad():
        main()

# BEVFusion deployment notes

Deep-dive notes for the BEVFusion deployment pipeline (PyTorch → ONNX sparse+dense → TensorRT →
evaluation). The architecture map is in the parent [`README.md`](../README.md).

| # | File | Topic |
|---|------|--------|
| 25 | [`25_README_AUTOWARE_COORD_CONTRACT_AND_EVAL_ALIGNMENT.md`](./25_README_AUTOWARE_COORD_CONTRACT_AND_EVAL_ALIGNMENT.md) | `coors` contract alignment with Autoware: why old/new ONNX both evaluate correctly, relation to ROS `x/y/z` and its boundaries |
| 26 | [`26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md`](./26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md) | Why `ScatterND -> SECOND` differs between `original` and split-merge ONNX; why separate tracing needs less shape-plumbing; numerical impact |
| 28 | [`28_README_BEVFUSION_2_8_DEPLOYMENT.md`](./28_README_BEVFUSION_2_8_DEPLOYMENT.md) | BEVFusion 2.8.x deployment notes |

Python entrypoints, configs, and pipelines live in the parent directory (`deployment/projects/bevfusion/`), alongside this `docs/` folder.

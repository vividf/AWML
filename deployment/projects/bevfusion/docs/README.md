# BEVFusion deployment notes

Numbered filenames follow **creation time** (filesystem birth time, with mtime then name as tie-break): **smaller number = older document**, **larger = newer** (latest context is toward the end of the list).

| # | File | Topic |
|---|------|--------|
| 1 | [`1_spconv_int8.md`](./1_spconv_int8.md) | spconv INT8 / libspconv alignment notes |
| 2 | [`2_spconv_int8_deploy.md`](./2_spconv_int8_deploy.md) | Recommended deploy flow for spconv INT8 |
| 3 | [`3_int8_implementation.md`](./3_int8_implementation.md) | Commands, config, Docker, error codes |
| 4 | [`4_spconv_int8_implementation_history_zh.md`](./4_spconv_int8_implementation_history_zh.md) | Implementation history (ZH), pitfalls |
| 5 | [`5_bevfusion_onnx_trt_spconv_int8.md`](./5_bevfusion_onnx_trt_spconv_int8.md) | ONNX / TensorRT / spconv INT8 overview |
| 6 | [`6_bevfusion_split_ptq_int8_progress.md`](./6_bevfusion_split_ptq_int8_progress.md) | Split ONNX + PTQ INT8 progress log |

Python entrypoints, configs, and pipelines live in the parent directory (`deployment/projects/bevfusion/`), alongside this `docs/` folder.

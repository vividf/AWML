# Copyright (c) OpenMMLab. All rights reserved.
"""Experimental / off-main-path BEVFusion tooling.

"Method 2" (libspconv engine) sparse-INT8 deployment: export the sparse encoder to a libspconv
INT8 ONNX (:mod:`.export_sparse_encoder_int8` + :mod:`.libspconv_onnx_exporter`) and benchmark it
against the main TensorRT-plugin route (``benchmark_sparse_int8.sh`` + ``cpp/``). This is a
documented alternative, not part of the live deploy pipeline; nothing in the main path imports it.
"""

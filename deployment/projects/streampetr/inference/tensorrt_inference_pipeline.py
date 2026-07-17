"""
StreamPETR TensorRT pipeline (engine-per-component inference).

Loads the three component engines and feeds each exactly the inputs its engine declares
(onnxsim pruned ``img_feats`` / ``data_timestamp`` before the engines were built). The
temporal memory queue is threaded on the host by the shared base class.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import tensorrt as trt
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.inference.gpu_resource_mixin import GPUResourceMixin, release_tensorrt_resources
from deployment.inference.tensorrt_runner import list_trt_io_names, load_trt_engine, run_trt_engine
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.projects.streampetr.inference.streampetr_inference_pipeline import (
    StreamPETRInferencePipeline,
)

logger = logging.getLogger(__name__)

_COMPONENT_NAMES = ("extract_img_feat", "position_embedding", "pts_head_memory")


class StreamPETRTensorRTInferencePipeline(GPUResourceMixin, StreamPETRInferencePipeline):
    """TensorRT-based StreamPETR pipeline."""

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        tensorrt_dir: str,
        device: DeviceSpec,
        components_cfg: ComponentsConfig,
    ) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.TENSORRT, device=device)

        self.tensorrt_dir = tensorrt_dir
        self._components_cfg = components_cfg
        self._engines: Dict[str, trt.ICudaEngine] = {}
        self._contexts: Dict[str, trt.IExecutionContext] = {}
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        self._load_tensorrt_engines()
        logger.info("TensorRT pipeline initialized with engines from: %s", tensorrt_dir)

    def _load_tensorrt_engines(self) -> None:
        trt.init_libnvinfer_plugins(self._trt_logger, "")
        runtime = trt.Runtime(self._trt_logger)
        for component_name in _COMPONENT_NAMES:
            engine_path = resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component_name=component_name,
                file_key="engine_file",
            )
            engine, context = load_trt_engine(runtime, engine_path, component_name=component_name)
            self._engines[component_name] = engine
            self._contexts[component_name] = context
            logger.info("Loaded TensorRT engine: %s", component_name)

    def _run(self, component: str, feeds: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        engine = self._engines[component]
        context = self._contexts[component]
        if context is None:
            raise RuntimeError(f"{component} context is None - likely failed to initialize due to GPU OOM")

        input_names, output_names = list_trt_io_names(engine)
        inputs_by_name = {}
        for name in input_names:
            if name not in feeds:
                raise KeyError(f"Engine '{component}' expects input '{name}' which was not provided")
            inputs_by_name[name] = self.to_numpy(feeds[name])

        outputs, _gpu_ms = run_trt_engine(engine, context, inputs_by_name, output_names)
        return outputs

    @override
    def run_encoder(self, img: torch.Tensor) -> torch.Tensor:
        outputs = self._run("extract_img_feat", {"img": img})
        return torch.from_numpy(outputs["img_feats"])

    @override
    def run_position_embedding(
        self,
        img_metas_pad: torch.Tensor,
        img_feats: torch.Tensor,
        intrinsics: torch.Tensor,
        img2lidar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self._run(
            "position_embedding",
            {
                "img_metas_pad": img_metas_pad,
                "img_feats": img_feats,
                "intrinsics": intrinsics,
                "img2lidar": img2lidar,
            },
        )
        return torch.from_numpy(outputs["pos_embed"]), torch.from_numpy(outputs["cone"])

    @override
    def run_head(self, head_inputs: Dict[str, torch.Tensor]) -> Sequence[torch.Tensor]:
        outputs = self._run("pts_head_memory", head_inputs)
        # Return in the configured (deployed) output order, not engine enumeration order.
        expected: List[str] = [out.name for out in self._components_cfg.get_component("pts_head_memory").io.outputs]
        return [torch.from_numpy(outputs[name]) for name in expected]

    def _release_gpu_resources(self) -> None:
        """Release TensorRT resources (engines and contexts)."""
        release_tensorrt_resources(engines=self._engines, contexts=self._contexts)

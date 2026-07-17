"""
StreamPETR ONNX Runtime pipeline (three sessions, one per component).

Feeds each session exactly the inputs its graph declares — onnxsim pruned the unused
``img_feats`` (position_embedding) and ``data_timestamp`` (pts_head_memory) tracing inputs,
so the sessions are the source of truth for input names.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.primitives.device import DeviceSpec
from deployment.projects.streampetr.inference.streampetr_inference_pipeline import (
    StreamPETRInferencePipeline,
)

logger = logging.getLogger(__name__)


class StreamPETRONNXInferencePipeline(StreamPETRInferencePipeline):
    """ONNX Runtime StreamPETR pipeline."""

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        onnx_dir: str,
        device: DeviceSpec,
        components_cfg: ComponentsConfig,
    ) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.ONNX, device=device)
        providers = device.to_ort_provider()
        self._sessions: Dict[str, ort.InferenceSession] = {}
        for name in ("extract_img_feat", "position_embedding", "pts_head_memory"):
            onnx_file = components_cfg.get_component(name).onnx_file
            path = os.path.join(onnx_dir, onnx_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"ONNX file for component '{name}' not found: {path}")
            self._sessions[name] = ort.InferenceSession(path, providers=providers)
            logger.info("Loaded ONNX session '%s' from %s", name, path)

    def _run(self, component: str, feeds: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        session = self._sessions[component]
        input_names = [inp.name for inp in session.get_inputs()]
        ort_feeds = {}
        for name in input_names:
            if name not in feeds:
                raise KeyError(f"Component '{component}' expects input '{name}' which was not provided")
            ort_feeds[name] = self.to_numpy(feeds[name])
        return session.run(None, ort_feeds)

    @override
    def run_encoder(self, img: torch.Tensor) -> torch.Tensor:
        (img_feats,) = self._run("extract_img_feat", {"img": img})
        return torch.from_numpy(img_feats)

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
        pos_embed, cone = outputs
        return torch.from_numpy(pos_embed), torch.from_numpy(cone)

    @override
    def run_head(self, head_inputs: Dict[str, torch.Tensor]) -> Sequence[torch.Tensor]:
        outputs = self._run("pts_head_memory", head_inputs)
        return [torch.from_numpy(out) for out in outputs]

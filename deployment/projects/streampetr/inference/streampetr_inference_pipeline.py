"""
StreamPETR inference pipeline base class.

Runs the deployed three-component flow per frame (encoder → position embedding → head) and
owns the **temporal memory queue** (migration spec §4.3, option A: pipeline-owned state).
The pipeline instance persists across `infer()` calls within an evaluation/verification run,
so the queue carries across frames exactly like the host runtime:

- queue reset on ``is_sequence_start`` (scene boundary) or first frame,
- ``pre_memory_* = stored[:, :memory_len]`` fed to the head, ``post_memory_*`` stored back,
- the timestamp arithmetic that the deployed graph intentionally leaves out (the commented
  lines in ``pts_head_onnx.py``) runs here on the host, in float64:
  ``pre = stored + t_frame`` before the head, ``stored = post - t_frame`` after — entries
  hold "age relative to the current frame".

Backends implement only the three raw stages (`run_encoder`, `run_position_embedding`,
`run_head`); preprocessing, state threading, and decoding are identical across backends.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.primitives.device import DeviceSpec
from deployment.projects.streampetr.io.sample_types import pad_to_4x4

logger = logging.getLogger(__name__)

#: Memory-queue field names, in the pre_/post_ tensor order of the deployed graph.
MEMORY_FIELDS = ("embedding", "reference_point", "timestamp", "egopose", "velo")


class StreamPETRInferencePipeline(BaseInferencePipeline):
    """Base pipeline for StreamPETR staged, stateful inference.

    Attributes:
        pytorch_model: Reference PyTorch model (config, head decode, memory geometry).
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        backend_type: Backend,
        device: DeviceSpec,
    ) -> None:
        super().__init__(model=pytorch_model, backend_type=backend_type, device=device)
        self.pytorch_model: torch.nn.Module = pytorch_model

        head = pytorch_model.pts_bbox_head
        self._memory_len = int(head.memory_len)
        self._embed_dims = int(head.embed_dims)
        # The decoder runs num_query learned queries PLUS num_propagated queries carried
        # over from the temporal memory (e.g. 644 + 256 = 900 rows per layer).
        self._num_query_total = int(head.num_query) + int(getattr(head, "num_propagated", 0))
        # Host-held queue: dict of float32 tensors (timestamp float64), or None before a clip.
        self._memory: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ state
    def reset_memory(self) -> None:
        """Drop the temporal queue; the next frame starts a fresh (zeroed) sequence."""
        self._memory = None

    def _zeroed_memory(self) -> Dict[str, torch.Tensor]:
        m, c = self._memory_len, self._embed_dims
        return {
            "embedding": torch.zeros(1, m, c),
            "reference_point": torch.zeros(1, m, 3),
            "timestamp": torch.zeros(1, m, 1, dtype=torch.float64),
            "egopose": torch.zeros(1, m, 4, 4),
            "velo": torch.zeros(1, m, 2),
        }

    # ------------------------------------------------------------------ stages
    @override
    def preprocess(self, input_data: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build the per-frame graph inputs from the loader's tensor dict.

        Args:
            input_data: Loader ``preprocess`` output (+ ``is_sequence_start`` injected by the
                executor); tensors as produced by ``StreamPETRDataset``.

        Returns:
            Dict with ``img``, ``intrinsics`` (4x4), ``img2lidar``, ``img_metas_pad``,
            ``data_ego_pose``, ``data_ego_pose_inv``, ``data_timestamp`` (float64) and the
            ``is_sequence_start`` flag.
        """
        img = input_data["img"].float()
        lidar2img = input_data["lidar2img"].float()
        pad_h, pad_w = int(img.shape[-2]), int(img.shape[-1])
        return {
            "img": img,
            "intrinsics": pad_to_4x4(input_data["intrinsics"].float()),
            "img2lidar": torch.inverse(lidar2img),
            "img_metas_pad": torch.tensor([pad_h, pad_w, 3], dtype=torch.float32),
            "data_ego_pose": input_data["ego_pose"].float(),
            "data_ego_pose_inv": input_data["ego_pose_inv"].float(),
            "data_timestamp": input_data["timestamp"].double().reshape(1),
            "is_sequence_start": bool(input_data.get("is_sequence_start", False)),
        }

    @override
    def run_model(
        self, preprocessed_input: Dict[str, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, float]]:
        """Run the three components, threading the memory queue across frames.

        Returns:
            ``((all_cls_scores, all_bbox_preds), stage_latencies)`` — head prediction tensors
            in the deployed layout (``[1, C, nb_dec*num_query]``).
        """
        stage_latencies: Dict[str, float] = {}

        if preprocessed_input["is_sequence_start"] or self._memory is None:
            if self._memory is not None:
                logger.debug("Sequence start: resetting temporal memory queue")
            self._memory = self._zeroed_memory()

        start = time.perf_counter()
        img_feats = self.run_encoder(preprocessed_input["img"])
        stage_latencies["encoder_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        pos_embed, cone = self.run_position_embedding(
            preprocessed_input["img_metas_pad"],
            img_feats,
            preprocessed_input["intrinsics"],
            preprocessed_input["img2lidar"],
        )
        stage_latencies["position_embedding_ms"] = (time.perf_counter() - start) * 1000

        # Host-side pre-update: entries become "age relative to the current frame".
        timestamp = preprocessed_input["data_timestamp"]
        pre_memory = dict(self._memory)
        pre_memory["timestamp"] = (pre_memory["timestamp"] + timestamp).float()

        head_inputs: Dict[str, torch.Tensor] = {
            "x": img_feats,
            "pos_embed": pos_embed,
            "cone": cone,
            "data_timestamp": timestamp,
            "data_ego_pose": preprocessed_input["data_ego_pose"],
            "data_ego_pose_inv": preprocessed_input["data_ego_pose_inv"],
        }
        for field in MEMORY_FIELDS:
            head_inputs[f"pre_memory_{field}"] = pre_memory[field]

        start = time.perf_counter()
        head_outputs = self.run_head(head_inputs)
        stage_latencies["head_ms"] = (time.perf_counter() - start) * 1000

        if len(head_outputs) < 7:
            raise RuntimeError(f"Expected >= 7 head outputs, got {len(head_outputs)}")
        all_cls_scores, all_bbox_preds = head_outputs[0], head_outputs[1]

        # Host-side post-update: slice back to memory_len (post_memory_* is memory_len + topk)
        # and remove the current frame time again.
        post_memory = dict(zip(MEMORY_FIELDS, head_outputs[2:7]))
        new_memory: Dict[str, torch.Tensor] = {}
        for field in MEMORY_FIELDS:
            tensor = torch.as_tensor(post_memory[field]).detach().cpu()[:, : self._memory_len]
            new_memory[field] = tensor.double() - timestamp if field == "timestamp" else tensor.float()
        self._memory = new_memory

        return (all_cls_scores, all_bbox_preds), stage_latencies

    @override
    def postprocess(
        self,
        model_output: Tuple[torch.Tensor, torch.Tensor],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Union[List[float], float, int]]]:
        """Decode head outputs into detection dicts via the model's own bbox coder.

        Reshapes the deployed layout ``[1, C, nb_dec*Q]`` back to ``[nb_dec, 1, Q, C]`` and
        calls ``StreamPETRHead.get_bboxes`` (NMS-free top-k + score threshold), so decode is
        identical across backends and to the training-time test path.
        """
        all_cls_scores = torch.as_tensor(model_output[0]).detach().cpu()
        all_bbox_preds = torch.as_tensor(model_output[1]).detach().cpu()

        def _unflatten(flat: torch.Tensor) -> torch.Tensor:
            # [1, C, nb_dec*Q] -> [1, nb_dec*Q, C] -> [nb_dec, 1, Q, C]
            channels = flat.shape[1]
            tokens = flat.shape[2]
            queries = self._num_query_total
            if tokens % queries != 0:
                raise RuntimeError(
                    f"Head output tokens ({tokens}) not divisible by query count ({queries}); "
                    "check num_query + num_propagated against the exported graph."
                )
            nb_dec = tokens // queries
            # contiguous(): the coder's decode_single calls .view(-1), which requires it.
            return flat.transpose(2, 1).reshape(nb_dec, 1, queries, channels).contiguous()

        preds_dicts = {
            "all_cls_scores": _unflatten(all_cls_scores),
            "all_bbox_preds": _unflatten(all_bbox_preds),
        }
        with torch.no_grad():
            bbox_list = self.pytorch_model.pts_bbox_head.get_bboxes(preds_dicts)

        results: List[Dict[str, Union[List[float], float, int]]] = []
        for bboxes, scores, labels in bbox_list:
            bboxes = bboxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            for i in range(len(bboxes)):
                results.append(
                    {
                        "bbox_3d": bboxes[i].tolist(),
                        "score": float(scores[i]),
                        "label": int(labels[i]),
                    }
                )
        return results

    # ------------------------------------------------------------------ backend hooks
    @abstractmethod
    def run_encoder(self, img: torch.Tensor) -> torch.Tensor:
        """Run the ``extract_img_feat`` component: ``img [1,N,3,H,W] -> img_feats``."""
        raise NotImplementedError

    @abstractmethod
    def run_position_embedding(
        self,
        img_metas_pad: torch.Tensor,
        img_feats: torch.Tensor,
        intrinsics: torch.Tensor,
        img2lidar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the ``position_embedding`` component: ``-> (pos_embed, cone)``."""
        raise NotImplementedError

    @abstractmethod
    def run_head(self, head_inputs: Dict[str, torch.Tensor]) -> Sequence[torch.Tensor]:
        """Run the ``pts_head_memory`` component.

        Args:
            head_inputs: Full traced-input dict (11 tensors). Backends whose graphs were
                pruned by onnxsim (ONNX/TensorRT drop ``data_timestamp``) feed only the
                tensors their runtime declares.

        Returns:
            The 14 head outputs in the deployed order (``all_cls_scores``,
            ``all_bbox_preds``, ``post_memory_*`` ×5, then the auxiliary outputs).
        """
        raise NotImplementedError

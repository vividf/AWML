"""YOLOX ONNX-export wrapper.

Reshapes the YOLOX head outputs into the Tier4 ONNX layout so the exported graph matches the
runtime's expectation and the PyTorch/ONNX/TensorRT backends stay byte-comparable. This is the
project's only export-time customization; everything else uses the shared whole-model export path
(``DefaultSampleExtractor`` + ``DefaultComponentBuilder``), so YOLOX needs no ``export/pipelines/``.
"""

from __future__ import annotations

import torch

from deployment.export.exporters.model_wrappers import BaseModelWrapper


class YOLOXONNXWrapper(BaseModelWrapper):
    """Wrap a YOLOX detector to emit the Tier4 ONNX output layout.

    Output: ``[batch, total_anchors, 4 + 1 + num_classes]`` = ``[bbox_reg(4), objectness(1),
    class_scores(num_classes)]``, where objectness and class scores are sigmoid-activated and the
    bbox regression is **raw** (decoded in the pipeline's postprocess, not in-graph). The input is
    ``[batch, 3, H, W]`` in BGR, range ``[0, 255]`` — the YOLOX ``data_preprocessor`` configures no
    mean/std, so the tensor is fed as-is, matching training.

    The static ``reshape`` + ``permute`` (rather than shape-derived ops) keeps the ONNX graph free of
    Shape/Gather/Unsqueeze nodes, matching the reference Tier4 YOLOX head export.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone+neck+head and concatenate per-level outputs into the Tier4 layout."""
        feat = self.model.extract_feat(x)
        cls_scores, bbox_preds, objectnesses = self.model.bbox_head(feat)

        outputs = []
        for cls_score, bbox_pred, objectness in zip(cls_scores, bbox_preds, objectnesses):
            outputs.append(torch.cat([bbox_pred, objectness.sigmoid(), cls_score.sigmoid()], 1))

        batch_size = outputs[0].shape[0]
        num_channels = outputs[0].shape[1]
        return torch.cat([o.reshape(batch_size, num_channels, -1) for o in outputs], dim=2).permute(0, 2, 1)

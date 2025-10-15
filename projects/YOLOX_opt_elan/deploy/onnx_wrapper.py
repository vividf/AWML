"""
YOLOX ONNX Export Wrapper - Tier4 Compatible Format.

This wrapper prepares YOLOX model for ONNX export with Tier4-compatible output format.

Output format: [batch_size, num_predictions, 4+1+num_classes]
where each prediction contains: [bbox_reg(4), objectness(1), class_scores(num_classes)]
"""

from typing import List

import torch
import torch.nn as nn


class YOLOXONNXWrapper(nn.Module):
    """
    Wrapper for YOLOX model to match Tier4 ONNX export format.

    The output format matches Tier4 YOLOX exactly:
    - Shape: [batch_size, total_anchors, 4 + 1 + num_classes]
    - Content: [bbox_predictions(4), objectness(1), class_scores(num_classes)]
    - objectness and class_scores are passed through sigmoid
    - bbox_predictions are raw regression outputs (NOT decoded)

    Args:
        model: MMDetection YOLOX model
        num_classes: Number of object classes (default: 8)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 8,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.bbox_head = model.bbox_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching Tier4 YOLOX format.

        Args:
            x: Input tensor [batch_size, 3, H, W]

        Returns:
            Concatenated predictions [batch_size, num_predictions, 4+1+num_classes]
            Format: [bbox_reg(4), objectness(1), class_scores(num_classes)]
            - bbox_reg: raw regression outputs (NOT decoded)
            - objectness: sigmoid activated
            - class_scores: sigmoid activated
        """
        # Extract features using model's method
        feat = self.model.extract_feat(x)

        # Get head outputs: (cls_scores, bbox_preds, objectnesses)
        cls_scores, bbox_preds, objectnesses = self.bbox_head(feat)

        # Process each detection level (matching Tier4 YOLOX logic)
        outputs = []

        for cls_score, bbox_pred, objectness in zip(cls_scores, bbox_preds, objectnesses):
            # Apply sigmoid to objectness and cls_score (NOT to bbox_pred)
            # This matches Tier4 YOLOX yolo_head.py line 198-200
            output = torch.cat([bbox_pred, objectness.sigmoid(), cls_score.sigmoid()], 1)
            outputs.append(output)

        # Flatten and concatenate all levels
        # This matches Tier4 YOLOX yolo_head.py line 218-220
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

        return outputs

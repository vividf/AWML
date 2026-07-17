"""Export-time ONNX graph definitions for the three StreamPETR components.

These modules are verbatim ports of the tracing containers in
``projects/StreamPETR/deploy/containers.py`` (the deployed-graph contract); see the
migration spec (§4.2) for the planned follow-up that replaces the hand-written head
forward with calls into the real ``StreamPETRHead`` methods.
"""

from deployment.projects.streampetr.export.onnx_models.encoder_onnx import StreamPETREncoderONNX
from deployment.projects.streampetr.export.onnx_models.position_embedding_onnx import (
    StreamPETRPositionEmbeddingONNX,
)
from deployment.projects.streampetr.export.onnx_models.pts_head_onnx import StreamPETRPtsHeadONNX

__all__ = [
    "StreamPETREncoderONNX",
    "StreamPETRPositionEmbeddingONNX",
    "StreamPETRPtsHeadONNX",
]

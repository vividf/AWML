"""YOLOX Deployment Module."""

from .data_loader import YOLOXDataLoader
from .evaluator import YOLOXEvaluator

__all__ = ["YOLOXDataLoader", "YOLOXEvaluator"]

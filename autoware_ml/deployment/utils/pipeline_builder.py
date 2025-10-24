"""
Utility for building preprocessing pipelines from model configs.

This module provides functions to extract and build test pipelines
from MMDet/MMDet3D/MMPretrain configs for use in deployment.
"""

import logging
from typing import Any, List

from mmengine.config import Config

logger = logging.getLogger(__name__)


def build_test_pipeline(model_cfg: Config, backend: str = "pytorch") -> Any:
    """
    Build test preprocessing pipeline from model config.

    This function extracts the test pipeline configuration from a model config
    and builds a Compose pipeline that can be used for preprocessing in deployment.

    Args:
        model_cfg: Model configuration containing test pipeline definition.
                   Can have pipeline defined in one of these locations:
                   - model_cfg.test_dataloader.dataset.pipeline
                   - model_cfg.test_pipeline
                   - model_cfg.val_dataloader.dataset.pipeline
        backend: Target backend ('pytorch', 'onnx', 'tensorrt').
                Currently not used, reserved for future backend-specific optimizations.

    Returns:
        Pipeline compose object (e.g., mmdet.datasets.transforms.Compose)

    Raises:
        ValueError: If no valid test pipeline found in config
        ImportError: If required transform packages are not available

    Examples:
        >>> from mmengine.config import Config
        >>> cfg = Config.fromfile('model_config.py')
        >>> pipeline = build_test_pipeline(cfg)
        >>> # Use pipeline to preprocess data
        >>> results = pipeline({'img_path': 'image.jpg'})
    """
    pipeline_cfg = _extract_pipeline_config(model_cfg)

    # Determine task type from config
    task_type = _infer_task_type(model_cfg, pipeline_cfg)


    # Build appropriate pipeline
    if task_type == "detection2d":
        return _build_detection2d_pipeline(pipeline_cfg)
    elif task_type == "detection3d":
        return _build_detection3d_pipeline(pipeline_cfg)
    elif task_type == "classification":
        return _build_classification_pipeline(pipeline_cfg)
    elif task_type == "segmentation":
        return _build_segmentation_pipeline(pipeline_cfg)
    else:
        # Fallback: try to build with mmdet
        logger.warning(f"Unknown task type '{task_type}', attempting to build with mmdet.Compose")
        return _build_detection2d_pipeline(pipeline_cfg)


def _extract_pipeline_config(model_cfg: Config) -> List:
    """
    Extract pipeline configuration from model config.

    Args:
        model_cfg: Model configuration

    Returns:
        List of transform configurations

    Raises:
        ValueError: If no valid pipeline found
    """
    # Try different possible locations for pipeline config
    pipeline_locations = [
        # Primary location: test_dataloader
        ("test_dataloader", "dataset", "pipeline"),
        # Alternative: direct test_pipeline
        ("test_pipeline",),
        # Fallback: val_dataloader
        ("val_dataloader", "dataset", "pipeline"),
    ]

    for location in pipeline_locations:
        try:
            cfg = model_cfg
            for key in location:
                cfg = cfg[key]
            if cfg:
                logger.info(f"Found test pipeline at: {'.'.join(location)}")
                return cfg
        except (KeyError, TypeError):
            continue

    raise ValueError(
        "No test pipeline found in config. "
        "Expected pipeline at one of: test_dataloader.dataset.pipeline, "
        "test_pipeline, or val_dataloader.dataset.pipeline"
    )


def _infer_task_type(model_cfg: Config, pipeline_cfg: List) -> str:
    """
    Infer task type from model config and pipeline.

    Args:
        model_cfg: Model configuration
        pipeline_cfg: Pipeline configuration

    Returns:
        Task type string ('detection2d', 'detection3d', 'classification', etc.)
    """
    # Check model type
    if "model" in model_cfg:
        model_type = model_cfg.model.get("type", "")
        model_type_lower = model_type.lower()

        # 3D detection models
        if any(
            name in model_type_lower
            for name in ["centerpoint", "pointpillars", "bevfusion", "transfusion", "streampetr"]
        ):
            return "detection3d"

        # 2D detection models
        if any(name in model_type_lower for name in ["yolo", "fasterrcnn", "retinanet", "fcos", "atss", "frnet"]):
            return "detection2d"

        # Classification models
        if any(name in model_type_lower for name in ["imageclassifier", "resnet", "mobilenet", "vit"]):
            return "classification"

        # Segmentation models
        if any(name in model_type_lower for name in ["segmentor", "unet", "fcn", "pspnet", "deeplabv3"]):
            return "segmentation"

    # Fallback: infer from pipeline transforms
    transform_types = [t.get("type", "") for t in pipeline_cfg]

    # 3D detection indicators
    if any("3D" in t or "Voxel" in t or "Point" in t for t in transform_types):
        return "detection3d"

    # Segmentation indicators
    if any("Seg" in t for t in transform_types):
        return "segmentation"

    # Classification indicators
    if any(t in ["PackClsInputs", "Normalize", "CenterCrop"] for t in transform_types):
        return "classification"

    # Default to 2D detection
    return "detection2d"


def _build_detection2d_pipeline(pipeline_cfg: List) -> Any:
    """
    Build 2D detection pipeline using mmdet transforms.

    Args:
        pipeline_cfg: List of transform configurations

    Returns:
        Compose object for mmdet transforms
    """
    # Try multiple import locations for Compose (different mmdet versions)
    import_errors = []

    # Try with init_default_scope (MMDet 3.x with mmengine)
    try:
        print("################## 1 ################")
        # Import mmdet first to register all transforms
        import mmdet.datasets.transforms  # noqa: F401
        from mmengine.dataset import Compose
        from mmengine.registry import init_default_scope

        # Set default scope to mmdet so Compose uses mmdet's TRANSFORMS registry
        init_default_scope("mmdet")
        logger.info("Building 2D detection pipeline with mmengine.dataset.Compose (default_scope='mmdet')")
        return Compose(pipeline_cfg)
    except (ImportError, Exception) as e:
        import_errors.append(f"mmengine.dataset.Compose with init_default_scope: {e}")



def _build_detection3d_pipeline(pipeline_cfg: List) -> Any:
    """
    Build 3D detection pipeline using mmdet3d transforms.

    Args:
        pipeline_cfg: List of transform configurations

    Returns:
        Compose object for mmdet3d transforms
    """
    # Try multiple import locations for Compose (different mmdet3d versions)
    import_errors = []

    # Try with init_default_scope (MMDet3D 1.x+ with mmengine)
    try:
        print("################## 4 ################")
        # Import mmdet3d first to register all transforms
        import mmdet3d.datasets.transforms  # noqa: F401
        from mmengine.dataset import Compose
        from mmengine.registry import init_default_scope

        # Set default scope to mmdet3d so Compose uses mmdet3d's TRANSFORMS registry
        init_default_scope("mmdet3d")
        logger.info("Building 3D detection pipeline with mmengine.dataset.Compose (default_scope='mmdet3d')")
        return Compose(pipeline_cfg)
    except (ImportError, Exception) as e:
        import_errors.append(f"mmengine.dataset.Compose with init_default_scope: {e}")


    # If all imports failed, raise error with details
    error_msg = "Failed to import Compose from any known location. Tried:\n" + "\n".join(
        f"  - {err}" for err in import_errors
    )
    error_msg += "\n\nPlease install mmdetection3d and mmcv/mmengine."
    raise ImportError(error_msg)

# TODO(vividf): this is not used for calibration classification yet
def _build_classification_pipeline(pipeline_cfg: List) -> Any:
    """
    Build classification pipeline using mmpretrain transforms.

    Args:
        pipeline_cfg: List of transform configurations

    Returns:
        Compose object for classification transforms
    """
    # Try multiple import locations for Compose (different versions)
    import_errors = []

    # Try mmpretrain with init_default_scope (newer versions)
    try:
        # Import mmpretrain first to register all transforms
        import mmpretrain.datasets.transforms  # noqa: F401
        from mmengine.dataset import Compose
        from mmengine.registry import init_default_scope

        # Set default scope to mmpretrain
        init_default_scope("mmpretrain")
        logger.info("Building classification pipeline with mmengine.dataset.Compose (default_scope='mmpretrain')")
        return Compose(pipeline_cfg)
    except (ImportError, Exception) as e:
        import_errors.append(f"mmengine.dataset.Compose with init_default_scope (mmpretrain): {e}")

    # Try mmcls with init_default_scope (older versions)
    try:
        # Import mmcls first to register all transforms
        import mmcls.datasets.pipelines  # noqa: F401
        from mmengine.dataset import Compose
        from mmengine.registry import init_default_scope

        # Set default scope to mmcls
        init_default_scope("mmcls")
        logger.info("Building classification pipeline with mmengine.dataset.Compose (default_scope='mmcls')")
        return Compose(pipeline_cfg)
    except (ImportError, Exception) as e:
        import_errors.append(f"mmengine.dataset.Compose with init_default_scope (mmcls): {e}")

    # Try without init_default_scope (fallback)
    try:
        # Try to import mmpretrain/mmcls first to register transforms
        try:
            import mmpretrain.datasets.transforms  # noqa: F401
        except ImportError:
            try:
                import mmcls.datasets.pipelines  # noqa: F401
            except ImportError:
                pass
        from mmcv.transforms import Compose

        logger.info("Building classification pipeline with mmcv.transforms.Compose (no default_scope)")
        return Compose(pipeline_cfg)
    except (ImportError, Exception) as e:
        import_errors.append(f"mmcv.transforms (no default_scope): {e}")


    # If all imports failed, raise error with details
    error_msg = "Failed to import Compose from any known location. Tried:\n" + "\n".join(
        f"  - {err}" for err in import_errors
    )
    error_msg += "\n\nPlease install mmpretrain/mmcls and mmcv/mmengine."
    raise ImportError(error_msg)


def _build_segmentation_pipeline(pipeline_cfg: List) -> Any:
    """
    Build segmentation pipeline using mmseg transforms.

    Args:
        pipeline_cfg: List of transform configurations

    Returns:
        Compose object for segmentation transforms
    """
    # Try multiple import locations for Compose (different versions)
    import_errors = []

    # Try mmseg with init_default_scope (newer versions)
    try:
        # Import mmseg first to register all transforms
        import mmseg.datasets.transforms  # noqa: F401
        from mmengine.dataset import Compose
        from mmengine.registry import init_default_scope

        # Set default scope to mmseg
        init_default_scope("mmseg")
        logger.info("Building segmentation pipeline with mmengine.dataset.Compose (default_scope='mmseg')")
        return Compose(pipeline_cfg)
    except (ImportError, Exception) as e:
        import_errors.append(f"mmengine.dataset.Compose with init_default_scope: {e}")


    # If all imports failed, raise error with details
    error_msg = "Failed to import Compose from any known location. Tried:\n" + "\n".join(
        f"  - {err}" for err in import_errors
    )
    error_msg += "\n\nPlease install mmsegmentation and mmcv/mmengine."
    raise ImportError(error_msg)


def validate_pipeline(pipeline: Any, sample_data: dict) -> bool:
    """
    Validate that a pipeline works correctly with sample data.

    Args:
        pipeline: Built pipeline to validate
        sample_data: Sample data dictionary to test with

    Returns:
        True if pipeline processes sample successfully

    Raises:
        RuntimeError: If pipeline validation fails
    """
    try:
        results = pipeline(sample_data)
        logger.info("Pipeline validation successful")
        return True
    except Exception as e:
        raise RuntimeError(f"Pipeline validation failed: {e}") from e

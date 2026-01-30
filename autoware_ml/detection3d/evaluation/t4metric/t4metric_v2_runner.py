"""
T4MetricV2 Runner - Evaluation runner for T4MetricV2.

This module provides the core evaluation logic that can be used by both
training (T4MetricV2 wrapper) and deployment (Detection3DMetricsInterface wrapper).
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore, MetricsScoreConfig
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.manager import PerceptionEvaluationManager
from pyquaternion import Quaternion

_UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrameInput:
    """Input frame data for evaluation runner."""

    scene_id: str
    sample_id: int  # Use int for internal identity (matching used_frame type)
    ground_truth_objects: FrameGroundTruth
    estimated_objects: List[ObjectType]
    frame_prefix: str
    unix_time: float


@dataclass(frozen=True)
class EvaluatorData:
    """Dataclass to save data related to a PerceptionEvaluationManager."""

    perception_evaluator_manager: PerceptionEvaluationManager
    bev_distance_range: Optional[Tuple[float, float]]
    perception_evaluator_configs: PerceptionEvaluationConfig
    frame_pass_fail_config: PerceptionPassFailConfig
    critical_object_filter_config: Optional[CriticalObjectFilterConfig]
    metric_score_config: MetricsScoreConfig
    min_range: float
    max_range: float
    range_filter_name: str


# ============================================================================
# Conversion Utilities
# ============================================================================


def labels_index_to_label(class_names: List[str], label_index: int) -> Label:
    """Convert a label index to a Label object.

    Args:
        class_names: List of class names.
        label_index: Index of the label in class_names.

    Returns:
        Label object with AutowareLabel.
    """
    if 0 <= label_index < len(class_names):
        class_name = class_names[label_index]
    else:
        class_name = _UNKNOWN

    autoware_label = AutowareLabel.__members__.get(class_name.upper(), AutowareLabel.UNKNOWN)
    return Label(label=autoware_label, name=class_name)


def bbox_to_dynamic_object(
    bbox: np.ndarray,
    label: Label,
    score: float,
    frame_id: str,
    unix_time: float,
    num_pts: Optional[int] = None,
) -> DynamicObject:
    """Convert a bounding box to a DynamicObject.

    Args:
        bbox: Bounding box array [x, y, z, l, w, h, yaw, vx, vy] or [x, y, z, l, w, h, yaw]
        label: Label object
        score: Confidence score (use 1.0 for ground truth)
        frame_id: Frame ID (e.g., "base_link")
        unix_time: Unix timestamp
        num_pts: Optional number of LiDAR points (for ground truth)

    Returns:
        DynamicObject instance.
    """
    x, y, z = bbox[0], bbox[1], bbox[2]
    l, w, h = bbox[3], bbox[4], bbox[5]
    yaw = bbox[6]

    # Velocity (optional)
    vx = bbox[7] if len(bbox) > 7 else 0.0
    vy = bbox[8] if len(bbox) > 8 else 0.0

    # Create quaternion from yaw
    orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

    kwargs = {
        "unix_time": unix_time,
        "frame_id": frame_id,
        "position": (x, y, z),
        "orientation": orientation,
        "shape": Shape(shape_type=ShapeType.BOUNDING_BOX, size=(l, w, h)),
        "velocity": (vx, vy, 0.0),
        "semantic_score": score,
        "semantic_label": label,
    }

    # Set pointcloud_num: use provided value, or default to 0 if None
    # This ensures compatibility with perception_eval which expects an integer
    kwargs["pointcloud_num"] = int(num_pts) if num_pts is not None else 0

    return DynamicObject(**kwargs)


# ============================================================================
# Evaluator Factory
# ============================================================================


def create_evaluators(
    perception_evaluator_configs: Dict[str, Any],
    frame_pass_fail_configs: Dict[str, Any],
    critical_object_filter_configs: Optional[Dict[str, Any]],
    target_labels: List[AutowareLabel],
    result_output_dir: str,
    write_metric_summary: bool = False,
) -> Dict[str, EvaluatorData]:
    """Create evaluators from configuration.

    Args:
        perception_evaluator_configs: Configuration dictionary for perception evaluation.
        frame_pass_fail_configs: Configuration dictionary for frame pass/fail criteria.
        critical_object_filter_configs: Optional configuration for critical object filtering.
        target_labels: List of target labels for evaluation.
        result_output_dir: Directory for saving evaluation results.
        write_metric_summary: Whether to write metric summary files.

    Returns:
        Dictionary mapping evaluator names to EvaluatorData.
    """
    # Overwrite result_output_dir
    perception_evaluator_configs = dict(perception_evaluator_configs)
    perception_evaluator_configs["result_root_directory"] = result_output_dir

    # Validate min_distance and max_distance
    assert (
        "min_distance" in perception_evaluator_configs["evaluation_config_dict"]
        and "max_distance" in perception_evaluator_configs["evaluation_config_dict"]
    ), "min_distance and max_distance must be provided in perception_evaluator_configs"

    assert isinstance(perception_evaluator_configs["evaluation_config_dict"]["min_distance"], list) and isinstance(
        perception_evaluator_configs["evaluation_config_dict"]["max_distance"], list
    ), (
        f"min_distance and max_distance must be a list, got: "
        f"{type(perception_evaluator_configs['evaluation_config_dict']['min_distance'])} and "
        f"{type(perception_evaluator_configs['evaluation_config_dict']['max_distance'])}"
    )

    # Form bev distance ranges
    bev_distance_ranges = []
    for min_distance, max_distance in zip(
        perception_evaluator_configs["evaluation_config_dict"]["min_distance"],
        perception_evaluator_configs["evaluation_config_dict"]["max_distance"],
    ):
        assert isinstance(min_distance, float) and isinstance(
            max_distance, float
        ), f"min_distance and max_distance must be floats, got: {type(min_distance)} and {type(max_distance)}"
        assert (
            min_distance < max_distance
        ), f"min_distance must be less than max_distance, got: {min_distance} and {max_distance}"
        bev_distance_ranges.append((min_distance, max_distance))

    range_filter_name = "bev_center"
    evaluators = {}
    for bev_distance_range in bev_distance_ranges:
        # Update min_distance and max_distance for this evaluator
        eval_config_dict = dict(perception_evaluator_configs["evaluation_config_dict"])
        eval_config_dict["min_distance"] = bev_distance_range[0]
        eval_config_dict["max_distance"] = bev_distance_range[1]

        evaluator_config_dict = dict(perception_evaluator_configs)
        evaluator_config_dict["evaluation_config_dict"] = eval_config_dict

        evaluator_config = PerceptionEvaluationConfig(**evaluator_config_dict)

        if critical_object_filter_configs is not None:
            perception_critical_object_filter_config = CriticalObjectFilterConfig(
                evaluator_config=evaluator_config,
                **critical_object_filter_configs,
            )
        else:
            perception_critical_object_filter_config = None

        perception_frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=evaluator_config,
            **frame_pass_fail_configs,
        )

        perception_metrics_score_config = MetricsScoreConfig(
            evaluator_config.evaluation_task, target_labels=target_labels
        )

        evaluator_name = f"{range_filter_name}_{bev_distance_range[0]}-{bev_distance_range[1]}"
        metric_output_dir = (
            str(Path(evaluator_config.visualization_directory) / evaluator_name) if write_metric_summary else None
        )

        evaluator = PerceptionEvaluationManager(
            evaluation_config=evaluator_config,
            load_ground_truth=False,
            metric_output_dir=metric_output_dir,
        )

        evaluators[evaluator_name] = EvaluatorData(
            perception_evaluator_manager=evaluator,
            bev_distance_range=bev_distance_range,
            perception_evaluator_configs=evaluator_config,
            frame_pass_fail_config=perception_frame_pass_fail_config,
            critical_object_filter_config=perception_critical_object_filter_config,
            metric_score_config=perception_metrics_score_config,
            min_range=bev_distance_range[0],
            max_range=bev_distance_range[1],
            range_filter_name=range_filter_name,
        )

    return evaluators


# ============================================================================
# Core Runner
# ============================================================================


class T4MetricV2Runner:
    """
    Pure evaluation runner for T4MetricV2.

    This runner handles:
    - Creating evaluators from configs
    - Processing frames sequentially or in parallel
    - Aggregating metrics
    - Exporting results

    It does NOT depend on mmengine or BaseMetric.
    """

    def __init__(
        self,
        perception_evaluator_configs: Dict[str, Any],
        frame_pass_fail_config: Dict[str, Any],
        critical_object_filter_config: Optional[Dict[str, Any]],
        target_labels: List[AutowareLabel],
        result_output_dir: str,
        write_metric_summary: bool = False,
        default_evaluator_prefix_name: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the T4MetricV2 runner.

        Args:
            perception_evaluator_configs: Configuration dictionary for perception evaluation.
            frame_pass_fail_config: Configuration dictionary for frame pass/fail criteria.
            critical_object_filter_config: Optional configuration for critical object filtering.
            target_labels: List of target labels for evaluation.
            result_output_dir: Directory for saving evaluation results.
            write_metric_summary: Whether to write metric summary files.
            default_evaluator_prefix_name: Optional prefix name for evaluators (e.g., "dataset_name/dataset_name").
            logger: Optional logger (e.g. MMLogger) for logging evaluator name and final metrics result.
        """
        self.logger = logger
        self.perception_evaluator_configs = perception_evaluator_configs
        self.frame_pass_fail_config = frame_pass_fail_config
        self.critical_object_filter_config = critical_object_filter_config
        self.target_labels = target_labels
        self.result_output_dir = result_output_dir
        self.write_metric_summary = write_metric_summary
        self.default_evaluator_prefix_name = default_evaluator_prefix_name or ""

        # Create evaluators
        self.evaluators = create_evaluators(
            perception_evaluator_configs=perception_evaluator_configs,
            frame_pass_fail_configs=frame_pass_fail_config,
            critical_object_filter_configs=critical_object_filter_config,
            target_labels=target_labels,
            result_output_dir=result_output_dir,
            write_metric_summary=write_metric_summary,
        )

        # The last evaluator is the main evaluator
        selected_evaluator_name = list(self.evaluators.keys())[-1]
        self.main_evaluator_name = (
            f"{self.default_evaluator_prefix_name}/{selected_evaluator_name}"
            if self.default_evaluator_prefix_name
            else selected_evaluator_name
        )
        self.main_evaluator_frame_id = self.evaluators[selected_evaluator_name].perception_evaluator_configs.frame_id

    def reset(self) -> None:
        """Reset all evaluator states to prevent metrics accumulation across multiple runs.

        This method clears the state of all PerceptionEvaluationManager instances,
        including frame_results and any internal caches. This is critical to prevent
        metrics from accumulating when compute_metrics() is called multiple times
        (e.g., in deployment when get_summary() calls compute_metrics() again).
        """
        for evaluator in self.evaluators.values():
            manager = evaluator.perception_evaluator_manager
            # Clear frame_results if it exists
            if hasattr(manager, "frame_results"):
                manager.frame_results.clear()
            # Clear any scene cache or internal state
            # Note: PerceptionEvaluationManager may have other internal state that needs clearing
            # This is a defensive approach - clear what we know about
            if hasattr(manager, "_scene_cache"):
                manager._scene_cache.clear()
            if hasattr(manager, "_used_frames"):
                manager._used_frames.clear()

    def run_sequential(self, frames: List[FrameInput]) -> Dict[str, Any]:
        """Process frames sequentially and return aggregated metrics.

        Args:
            frames: List of FrameInput instances.

        Returns:
            Dictionary containing:
                - aggregated_metric_scalars: Dict[str, Dict[str, float]]
                - aggregated_metric_data: Dict[str, Dict[str, Any]]
                - scene_metrics: Optional[Dict] (if write_metric_summary)
        """
        # Reset evaluator states to prevent metrics accumulation
        self.reset()

        # Sort frames by (scene_id, sample_id) to ensure correct previous frame tracking
        # This is critical for temporal metrics (tracking, temporal pass-fail, etc.)
        frames = sorted(frames, key=lambda f: (f.scene_id, f.sample_id))

        # Maintain previous frame result per (evaluator_name, scene_id)
        # Key: (evaluator_name, scene_id), Value: PerceptionFrameResult
        previous_frame_results: Dict[Tuple[str, str], PerceptionFrameResult] = {}

        # Process all frames
        for frame in frames:
            for evaluator_name, evaluator in self.evaluators.items():
                try:
                    # Validate frame data before processing
                    if frame.unix_time is None:
                        raise ValueError(f"unix_time is None for frame {frame.scene_id}/{frame.sample_id}")
                    if frame.ground_truth_objects is None:
                        raise ValueError(f"ground_truth_objects is None for frame {frame.scene_id}/{frame.sample_id}")
                    if frame.estimated_objects is None:
                        raise ValueError(f"estimated_objects is None for frame {frame.scene_id}/{frame.sample_id}")

                    manager = evaluator.perception_evaluator_manager

                    # Step 1: Preprocess current frame (equivalent to original preprocess_object_results)
                    # Use ground_truth_objects.unix_time (int, microseconds) instead of frame.unix_time (float)
                    # to match the expected type for preprocess_object_results
                    current_perception_frame_result = manager.preprocess_object_results(
                        unix_time=frame.ground_truth_objects.unix_time,
                        ground_truth_now_frame=frame.ground_truth_objects,
                        estimated_objects=frame.estimated_objects,
                        frame_pass_fail_config=evaluator.frame_pass_fail_config,
                        critical_object_filter_config=evaluator.critical_object_filter_config,
                    )

                    # Step 2: Get previous frame result for this (evaluator_name, scene_id)
                    key = (evaluator_name, frame.scene_id)
                    previous_perception_frame_result = previous_frame_results.get(key, None)

                    # Step 3: Evaluate current frame with explicit previous frame
                    # (equivalent to original evaluate_perception_frame(current, previous))
                    manager.evaluate_perception_frame(
                        perception_frame_result=current_perception_frame_result,
                        previous_perception_frame_result=previous_perception_frame_result,
                    )

                    # Step 4: Append to frame_results manually
                    manager.frame_results.append(current_perception_frame_result)

                    # Step 5: Update previous frame result for this (evaluator_name, scene_id)
                    previous_frame_results[key] = current_perception_frame_result

                except Exception as e:
                    # Log warning with more details but continue
                    import logging
                    import traceback

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to process frame {frame.scene_id}/{frame.sample_id} "
                        f"with evaluator {evaluator_name}: {e}"
                    )
                    logger.debug(f"Traceback: {traceback.format_exc()}")

        # Aggregate metrics
        return self._aggregate_metrics(frames)

    def _aggregate_metrics(self, frames: List[FrameInput]) -> Dict[str, Any]:
        """Aggregate metrics from all evaluators.

        Args:
            frames: List of processed frames (for building sample_id to prefix mapping).

        Returns:
            Dictionary containing aggregated metrics.
        """
        # Build sample_id to prefix mapping
        sample_id_to_prefix_frame_mapping: Dict[int, str] = {frame.sample_id: frame.frame_prefix for frame in frames}

        aggregated_metric_scalars = {}
        aggregated_metric_data = {}

        for evaluator_name, evaluator in self.evaluators.items():
            evaluator_full_name = (
                f"{self.default_evaluator_prefix_name}/{evaluator_name}"
                if self.default_evaluator_prefix_name
                else evaluator_name
            )

            # Get scene result
            final_metric_score = evaluator.perception_evaluator_manager.get_scene_result()

            # Log evaluator name and final metrics result (restored from pre-refactor _process_evaluator_results)
            if self.logger is not None:
                self.logger.info(f"====Evaluator: {evaluator_full_name}====")
                self.logger.info(f"Final metrics result: \n{final_metric_score}")

            # Process scalar metrics
            aggregated_metric_scalars[evaluator_full_name] = self._process_metrics_for_aggregation(
                final_metric_score, evaluator_name, sample_id_to_prefix_frame_mapping, evaluator
            )

            # Process metric data (precisions/recalls)
            aggregated_metric_data[evaluator_full_name] = self._aggregate_metrics_data(final_metric_score)

            # Process prefix-based aggregation
            frame_prefix_scores = evaluator.perception_evaluator_manager.get_scene_result_with_prefix()
            for frame_prefix_name, metric_dict in frame_prefix_scores.items():
                evaluator_frame_prefix_name = f"{frame_prefix_name}/{evaluator_name}"
                aggregated_metric_scalars[evaluator_frame_prefix_name] = self._process_metrics_for_aggregation(
                    metric_dict, evaluator_name, sample_id_to_prefix_frame_mapping, evaluator
                )
                aggregated_metric_data[evaluator_frame_prefix_name] = self._aggregate_metrics_data(metric_dict)

        result = {
            "aggregated_metric_scalars": aggregated_metric_scalars,
            "aggregated_metric_data": aggregated_metric_data,
        }

        # Optionally include scene metrics
        if self.write_metric_summary:
            result["scene_metrics"] = self._build_scene_metrics(frames)

        return result

    def _process_metrics_for_aggregation(
        self,
        metrics_score: MetricsScore,
        evaluator_name: str,
        sample_id_to_prefix_frame_mapping: Dict[int, str],
        evaluator: EvaluatorData,
    ) -> Dict[str, float]:
        """Process metrics from MetricsScore into a dictionary.

        Args:
            metrics_score: MetricsScore instance.
            evaluator_name: Name of the evaluator.
            sample_id_to_prefix_frame_mapping: Mapping from sample_id to frame_prefix.
            evaluator: EvaluatorData instance.

        Returns:
            Dictionary of processed metrics.
        """
        metric_dict = {}

        total_num_preds = 0
        for map_instance in metrics_score.mean_ap_values:
            num_preds = 0
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            # Process individual AP values
            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value

                label_num_preds = aps[0].objects_results_num if len(aps) else 0
                label_num_gts = map_instance.num_ground_truth_dict.get(label, 0) if len(aps) else 0
                num_preds += label_num_preds

                for ap in aps:
                    threshold = ap.matching_threshold
                    ap_value = ap.ap

                    # Create the metric key (unified naming)
                    key = f"T4MetricV2_label/{label_name}_AP_{matching_mode}_{threshold}"
                    metric_dict[key] = ap_value

                    # Max F1 score
                    metric_dict[f"T4MetricV2_label/{label_name}_max-f1score_{matching_mode}_{threshold}"] = (
                        ap.max_f1_score
                    )

                    # Optimal confidence threshold
                    metric_dict[f"T4MetricV2_label/{label_name}_optimal-confidence_{matching_mode}_{threshold}"] = (
                        ap.optimal_conf
                    )

                    # Optimal recall and precision
                    metric_dict[f"T4MetricV2_label/{label_name}_optimal-recall_{matching_mode}_{threshold}"] = (
                        ap.optimal_recall
                    )
                    metric_dict[f"T4MetricV2_label/{label_name}_optimal-precision_{matching_mode}_{threshold}"] = (
                        ap.optimal_precision
                    )

                # Label metadata
                metric_dict[f"metadata_label/test_{label_name}_num_predictions"] = label_num_preds
                metric_dict[f"metadata_label/test_{label_name}_num_ground_truths"] = label_num_gts

            # Add mAP and mAPH values
            map_key = f"T4MetricV2/mAP_{matching_mode}"
            maph_key = f"T4MetricV2/mAPH_{matching_mode}"
            metric_dict[map_key] = map_instance.map
            metric_dict[maph_key] = map_instance.maph

            total_num_preds = num_preds

        # Add metadata information
        metric_dict["metadata/test_min_range"] = evaluator.min_range
        metric_dict["metadata/test_max_range"] = evaluator.max_range
        metric_dict["metadata/test_range_filter_name"] = evaluator.range_filter_name
        metric_dict["metadata/test_total_num_frames"] = metrics_score.num_frame
        metric_dict["metadata/test_total_num_ground_truths"] = metrics_score.num_ground_truth
        metric_dict["metadata/test_total_num_predictions"] = total_num_preds

        # Add frame distribution by prefix
        # used_frame is List[int] (from MetricsScore), matching sample_id type (int)
        # Direct matching without conversion
        test_num_frame_distribution = defaultdict(int)
        for used_frame in metrics_score.used_frame:
            test_num_frame_distribution[sample_id_to_prefix_frame_mapping.get(used_frame, "unknown")] += 1

        metric_dict["metadata/test_num_frame_distribution"] = dict(test_num_frame_distribution)

        return metric_dict

    def _aggregate_metrics_data(self, metrics_score: MetricsScore) -> Dict[str, Any]:
        """Process iterable metrics (precisions/recalls) from MetricsScore.

        Args:
            metrics_score: MetricsScore instance.

        Returns:
            Dictionary containing iterable metrics.
        """
        iterable_metrics = {}

        for map_instance in metrics_score.mean_ap_values:
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            # Process individual AP values
            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value

                for ap in aps:
                    threshold = ap.matching_threshold

                    # Create precision_interpolate and recall_interpolate keys
                    iterable_metrics[
                        f"T4MetricV2_label_detection/{label_name}_precisions_{matching_mode}_{threshold}"
                    ] = ap.precision_interp.tolist()
                    iterable_metrics[
                        f"T4MetricV2_label_detection/{label_name}_recalls_{matching_mode}_{threshold}"
                    ] = ap.recall_interp.tolist()

        return iterable_metrics

    def _build_scene_metrics(self, frames: List[FrameInput]) -> Dict[str, Any]:
        """Build scene-level metrics structure.

        Args:
            frames: List of processed frames.

        Returns:
            Dictionary of scene metrics.
        """
        # Group frames by scene_id
        scenes = defaultdict(lambda: defaultdict(dict))
        for frame in frames:
            scenes[frame.scene_id][frame.sample_id] = frame

        scene_metrics = {
            scene_id: {sample_id: {} for sample_id in samples.keys()} for scene_id, samples in scenes.items()
        }

        # Populate scene metrics from evaluator results
        for evaluator_name, evaluator in self.evaluators.items():
            for frame_result in evaluator.perception_evaluator_manager.frame_results:
                # Extract scene_id and sample_id from frame_name
                # frame_name is str (from FrameGroundTruth, pure digit string like "123")
                # Convert to int to match sample_id type (int)
                try:
                    frame_name_int = int(frame_result.frame_name)
                except (ValueError, TypeError):
                    # Skip if frame_name cannot be converted to int
                    continue
                # Try to find matching frame
                for scene_id, samples in scenes.items():
                    if frame_name_int in samples:
                        sample_id = frame_name_int
                        frame_prefix = samples[frame_name_int].frame_prefix
                        frame_metrics = scene_metrics[scene_id][sample_id].setdefault(
                            f"{frame_prefix}/{evaluator_name}", {}
                        )

                        # Process AP values
                        for map_instance in frame_result.metrics_score.mean_ap_values:
                            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")
                            matching_metrics = frame_metrics.setdefault(matching_mode, {})

                            # Process AP values
                            for label, aps in map_instance.label_to_aps.items():
                                label_name = label.value
                                label_metrics = matching_metrics.setdefault(label_name, {})
                                ap_metrics = label_metrics.setdefault("ap", {})
                                for ap in aps:
                                    threshold_str = str(ap.matching_threshold)
                                    ap_metrics[threshold_str] = ap.ap

                            # Process APH values
                            label_to_aphs = getattr(map_instance, "label_to_aphs", None)
                            if label_to_aphs:
                                for label, aphs in label_to_aphs.items():
                                    label_name = label.value
                                    label_metrics = matching_metrics.setdefault(label_name, {})
                                    aph_metrics = label_metrics.setdefault("aph", {})
                                    for aph in aphs:
                                        threshold_str = str(aph.matching_threshold)
                                        aph_metrics[threshold_str] = aph.ap

        return dict(scene_metrics)

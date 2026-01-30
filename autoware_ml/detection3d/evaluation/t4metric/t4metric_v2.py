import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.dist import get_world_size
from mmengine.evaluator import BaseMetric
from mmengine.logging import MessageHub, MMLogger
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.result.perception_frame import PerceptionFrame

from autoware_ml.detection3d.evaluation.t4metric.t4metric_v2_dataframe import T4MetricV2DataFrame
from autoware_ml.detection3d.evaluation.t4metric.t4metric_v2_runner import (
    FrameInput,
    T4MetricV2Runner,
    bbox_to_dynamic_object,
    labels_index_to_label,
)

__all__ = ["T4MetricV2"]
_UNKNOWN = "unknown"
DEFAULT_T4METRIC_FILE_NAME = "t4metric_v2_results_{}.pkl"
DEFAULT_T4METRIC_METRICS_FOLDER = "metrics"
DEFAULT_T4METRIC_RESULT_FOLDER = "result"


@METRICS.register_module()
class T4MetricV2(BaseMetric):
    """T4 format evaluation metric V2.
    Args:
        data_root (str):
            Path of dataset root.
        ann_file (str):
            Path of annotation file.
        dataset_name (str): Dataset running metrics.
        output_dir (str): Directory to save the evaluation results. Note that it's working_directory/<output_dir>.
        write_metric_summary (bool): Whether to write metric summary to json files.
        prefix (str, optional):
            The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        collect_device (str):
            Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or 'gpu'.
            Defaults to 'cpu'.
        class_names (List[str], optional):
            The class names. Defaults to [].
        name_mapping (dict, optional):
            The data class mapping, applied to ground truth during evaluation.
            Defaults to None.
        perception_evaluator_configs (Dict[str, Any]):
            Configuration dictionary for perception evaluation.
        critical_object_filter_config (Dict[str, Any]):
            Configuration dictionary for filtering critical objects during evaluation.
        frame_pass_fail_config (Dict[str, Any]):
            Configuration dictionary that defines pass/fail criteria for perception evaluation.
        results_pickle_path (Optional[Union[Path, str]]):
            Path to the pickle file used for saving or loading prediction and ground truth results.

            - If not provided: runs `process()` and `compute_metrics()`.
            - If provided but the file does not exist: runs `process()` and `compute_metrics()`,
              then saves predictions and ground truth to the given path.
            - If provided and the file exists: skips `process()`, loads predictions and
              ground truth from the pickle file, and runs `compute_metrics()`.

            Defaults to None.
      bev_distance_ranges (Optional[Tuple[float]]):
        Bev distance ranges in meters for different range buckets. Defaults to None.
        Example: [(0.0, 60.0), (60.0, 90.0), (90.0, 121.0), (0.0, 121.0)], which means it will compute the metrics
        for bev distance ranges are [0.0, 60.0), [60.0, 90.0), [90.0, 121.0), [0.0, 121.0) after filtering objects by bev distance ranges, respectively.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        training_statistics_parquet_path: str,
        validation_statistics_parquet_path: str,
        dataset_name: str,
        output_dir: str,
        experiment_name: str,
        experiment_group_name: str,
        write_metric_summary: bool,
        checkpoint_path: Optional[Union[Path, str]] = None,
        scene_batch_size: int = 128,
        num_workers: int = 8,
        prefix: Optional[str] = None,
        collect_device: str = "cpu",
        class_names: List[str] = None,
        name_mapping: Optional[dict] = None,
        perception_evaluator_configs: Optional[Dict[str, Any]] = None,
        critical_object_filter_config: Optional[Dict[str, Any]] = None,
        frame_pass_fail_config: Optional[Dict[str, Any]] = None,
        results_pickle_path: Optional[Union[Path, str]] = None,
    ) -> None:
        self.default_prefix = "T4MetricV2"
        self.dataset_name = dataset_name
        super(T4MetricV2, self).__init__(collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        self.data_root = data_root
        self.num_workers = num_workers
        self.scene_batch_size = scene_batch_size
        self.class_names = class_names
        self.experiment_name = experiment_name
        self.experiment_group_name = experiment_group_name
        self.name_mapping = name_mapping
        if name_mapping is not None:
            self.class_names = [self.name_mapping.get(name, name) for name in self.class_names]

        self.target_labels = [AutowareLabel[label.upper()] for label in self.class_names]

        # scene_id to index map in self.results
        self.scene_id_to_index_map: Dict[str, int] = {}

        self.message_hub = MessageHub.get_current_instance()
        self.logger = MMLogger.get_current_instance()
        self.logger_file_path = Path(self.logger.log_file).parent
        self.test_timestamp = self.logger_file_path.parts[-1]
        self.checkpoint_path = checkpoint_path

        # Set output directory for metrics files
        assert output_dir, f"output_dir must be provided, got: {output_dir}"
        self.output_dir = self.logger_file_path / output_dir / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Metrics output directory set to: {self.output_dir}")

        self.results_pickle_path: Optional[Path] = results_pickle_path
        if self.results_pickle_path and self.results_pickle_path.suffix != ".pkl":
            raise ValueError(f"results_pickle_path must end with '.pkl', got: {self.results_pickle_path}")
        self.results_pickle_exists = True if self.results_pickle_path and self.results_pickle_path.exists() else False

        self.write_metric_summary = write_metric_summary
        self.num_running_gpus = get_world_size()
        self.result_output_dir = self.output_dir / DEFAULT_T4METRIC_RESULT_FOLDER
        self.default_evaluator_prefix_name = f"{dataset_name}/{dataset_name}"

        # Create runner (pure evaluation logic)
        self.runner = T4MetricV2Runner(
            perception_evaluator_configs=perception_evaluator_configs,
            frame_pass_fail_config=frame_pass_fail_config,
            critical_object_filter_config=critical_object_filter_config,
            target_labels=self.target_labels,
            result_output_dir=str(self.result_output_dir),
            write_metric_summary=write_metric_summary,
            default_evaluator_prefix_name=self.default_evaluator_prefix_name,
            logger=self.logger,
        )

        # The last evaluator is the main evaluator
        self.main_evaluator_name = self.runner.main_evaluator_name
        self.main_evaluator_frame_id = self.runner.main_evaluator_frame_id
        self.logger.info(f"{self.default_prefix} running with {self.num_running_gpus} GPUs")

        # T4MetricV2 DataFrame
        self.t4metric_v2_dataframe_output_path = self.output_dir / f"t4metricv2_metrics_{self.test_timestamp}.parquet"
        self.t4_metric_v2_dataframe = T4MetricV2DataFrame(
            output_dataframe_path=self.t4metric_v2_dataframe_output_path,
            training_statistics_parquet_path=Path(training_statistics_parquet_path),
            validation_statistics_parquet_path=Path(validation_statistics_parquet_path),
        )

    def evaluate(self, size: int) -> Dict[str, float]:
        """
        Evaluate the results and return a dict of metrics. Override of BaseMetric.evaluate to clean up caches
        for the multi-gpu case.
        """
        metrics = super().evaluate(size=size)
        # Clean up any caches for multi-gpu case
        self._clean_up()

        return metrics

    # override of BaseMetric.process
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model and the ground truth of dataset.
        """

        if self.results_pickle_exists:
            # Skip processing if result pickle already exists
            return

        for data_sample in data_samples:
            current_time = data_sample["timestamp"]
            scene_id = self._parse_scene_id(data_sample["lidar_path"])
            # Ensure sample_idx is int (matching FrameInput.sample_id type)
            sample_idx_raw = data_sample.get("sample_idx", None)
            if sample_idx_raw is None:
                raise ValueError("sample_idx is required in data_sample")
            sample_id_int = int(sample_idx_raw)  # Guarantee it's int
            frame_ground_truth = self._parse_ground_truth_from_sample(current_time, data_sample)
            perception_frame = self._parse_predictions_from_sample(current_time, data_sample, frame_ground_truth)
            self._save_perception_frame(scene_id, sample_id_int, perception_frame)

    # override of BaseMetric.compute_metrics
    def compute_metrics(
        self,
        results: List[dict],
    ) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        Example:
            Metric dictionary:
            {
                'T4MetricV2/car_AP_center_distance_0.5': 0.7
                'T4MetricV2/truck_AP_center_distance_0.5': 0.7,
                'T4MetricV2/bus_AP_center_distance_0.5': 0.7,
                'T4MetricV2/bicycle_AP_center_distance_0.5': 0.7,
                'T4MetricV2/pedestrian_AP_center_distance_0.5': 0.7,
                ...
            }
        """
        try:
            # Load or save results based on pickle configuration
            results = self._handle_results_persistence(results)
            # Validate input
            self._validate_results(results)

            # Initialize scenes
            scenes = {scene_id: samples for scene in results for scene_id, samples in scene.items()}

            # Convert scenes to FrameInput list for runner
            frames = []
            for scene_id, samples in scenes.items():
                for sample_id, perception_frame in samples.items():
                    # Ensure sample_id is int (matching FrameInput.sample_id type)
                    # Convert from str if needed (for backward compatibility with old pickle files)
                    if isinstance(sample_id, str):
                        sample_id_int = int(sample_id)
                    elif isinstance(sample_id, int):
                        sample_id_int = sample_id
                    else:
                        raise TypeError(
                            f"sample_id must be int or str (convertible to int), got {type(sample_id)}: {sample_id}"
                        )
                    frames.append(
                        FrameInput(
                            scene_id=scene_id,
                            sample_id=sample_id_int,
                            ground_truth_objects=perception_frame.ground_truth_objects,
                            estimated_objects=perception_frame.estimated_objects,
                            frame_prefix=perception_frame.ground_truth_objects.frame_prefix,
                            unix_time=perception_frame.unix_time,
                        )
                    )

            # Run runner to compute metrics
            runner_results = self.runner.run_sequential(frames)

            # Add training-specific metadata to all evaluator results (required for DataFrame column consistency)
            for _evaluator_name, evaluator_metrics in runner_results["aggregated_metric_scalars"].items():
                evaluator_metrics["metadata/experiment_name"] = self.experiment_name
                evaluator_metrics["metadata/experiment_group_name"] = self.experiment_group_name
                evaluator_metrics["metadata/test_timestamp"] = self.test_timestamp
                evaluator_metrics["metadata/test_checkpoint_path"] = self.checkpoint_path
                evaluator_metrics["metadata/test_dataset_name"] = self.dataset_name

            main_metrics = runner_results["aggregated_metric_scalars"][self.main_evaluator_name]

            # Write aggregated metrics for all evaluators to an output file
            if self.write_metric_summary:
                try:
                    # Write scene metrics if available
                    if "scene_metrics" in runner_results:
                        self._write_scene_metrics_from_dict(runner_results["scene_metrics"])

                    metric_scalars_json = self._write_aggregated_metrics(
                        runner_results["aggregated_metric_scalars"], "aggregated_metrics.json"
                    )
                    metric_data_json = self._write_aggregated_metrics(
                        runner_results["aggregated_metric_data"], "aggregated_metrics_data.json"
                    )

                    # Write to a parquet
                    df = self.t4_metric_v2_dataframe(
                        aggregated_metric_scalars=metric_scalars_json, aggregated_metric_data=metric_data_json
                    )
                    self.t4_metric_v2_dataframe.save_dataframe(df)
                    self.logger.info(
                        f"Saved aggregated metrics to a parquet file: {self.t4_metric_v2_dataframe.output_dataframe_path}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to write aggregated metrics to output files: {e}")

            # Return metrics from the main evaluator
            return main_metrics

        except Exception as e:
            raise RuntimeError(f"Error in compute_metrics: {e}")
        finally:
            self._clean_up()

    def _validate_results(self, results: List[dict]) -> None:
        """Validate that the results contain valid data.

        Args:
            results (List[dict]): The results to validate.

        Raises:
            ValueError: If results are invalid.
        """
        assert results, "Results list is empty"

        assert isinstance(results, list), f"Results must be a list, got {type(results)}"

        # Check that each result is a dictionary
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                raise ValueError(f"Result at index {i} must be a dictionary, got {type(result)}")

            # Check that each result contains scene data
            if not result:
                raise ValueError(f"Result at index {i} is empty")

        self.logger.info(f"Validated {len(results)} scenes")

    def _collate_results(self, results: List[dict]) -> List[dict]:
        """Collate results from multiple GPUs.

        Args:
            results (List[dict]): List of results from different GPUs.

        Returns:
        """
        # Reinitialize
        self.scene_id_to_index_map: Dict[str, int] = {}

        # [{scene_id: {sample_id: perception_frame}}]
        tmp_results = []
        for scenes in results:
            for scene_id, samples in scenes.items():
                result_index = self.scene_id_to_index_map.get(scene_id, None)
                if result_index is not None:
                    tmp_results[result_index][scene_id].update(samples)
                else:
                    self.scene_id_to_index_map[scene_id] = len(tmp_results)
                    tmp_results.append({scene_id: samples})

        # Reorder all samples in all scenes
        for result in tmp_results:
            for scene_id, samples in result.items():
                result[scene_id] = {k: v for k, v in sorted(samples.items(), key=lambda item: item[0])}

        # Update results to the collated results
        self.results = tmp_results
        self.logger.info(f"Collated results from {len(results)} into {len(self.results)} scenes")
        return tmp_results

    def _handle_results_persistence(self, results: List[dict]) -> List[dict]:
        """Handle loading or saving results based on pickle configuration.

        Args:
            results (List[dict]): The current results.

        Returns:
            List[dict]: The results to use for evaluation.
        """
        if self.results_pickle_exists:
            self.logger.info(f"Loading results from pickle file: {self.results_pickle_path}")
            with open(self.results_pickle_path, "rb") as f:
                results = pickle.load(f)

            return results

        # Reorganize results from multi-gpu
        if self.num_running_gpus > 1:
            results = self._collate_results(results)

        # Save results to a pickle file
        current_epoch = self.message_hub.get_info("epoch", -1) + 1
        results_output_path = self.result_output_dir / DEFAULT_T4METRIC_FILE_NAME.format(current_epoch)
        self.logger.info(f"Saving results of epoch: {current_epoch} to pickle file: {results_output_path}")

        # Create parent directory if needed
        results_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_output_path, "wb") as f:
            pickle.dump(self.results, f)

        return results

    def _clean_up(self) -> None:
        """Clean up resources after computation."""
        self.scene_id_to_index_map.clear()
        # Reset runner evaluator states to prevent state accumulation
        # This is important for multi-GPU scenarios and when compute_metrics() is called multiple times
        self.runner.reset()

    def _write_scene_metrics_from_dict(self, scene_metrics: Dict[str, Any]) -> None:
        """Write scene metrics from a dictionary.

        Writes one file per evaluator; each file contains only that evaluator's
        scene metrics (matching pre-refactor _write_scene_metrics behavior).

        Args:
            scene_metrics: Dictionary of scene metrics. Structure is
                scene_id -> sample_id -> "{frame_prefix}/{evaluator_name}" -> metrics.
        """
        suffix = "/"
        for evaluator_name in self.runner.evaluators.keys():
            try:
                # Filter to only this evaluator's keys (key format: frame_prefix/evaluator_name)
                filtered = {}
                for scene_id, samples in scene_metrics.items():
                    filtered[scene_id] = {}
                    for sample_id, frame_dict in samples.items():
                        filtered[scene_id][sample_id] = {
                            k: v for k, v in frame_dict.items() if k.endswith(suffix + evaluator_name)
                        }
                output_path = self.output_dir / evaluator_name / "scene_metrics.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as scene_file:
                    json.dump(filtered, scene_file, indent=4)
                self.logger.info(f"Scene metrics written to: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to write scene metrics to output files: {e}")

    def _write_aggregated_metrics(
        self, final_metric_dict: dict, aggregated_metric_file_name: str = "aggregated_metrics.json"
    ) -> Dict[str, Any]:
        """
        Writes aggregated metrics to a JSON file with the specified format.

        Args:
            final_metric_dict {evaluator_name: {metric_name: metric_value}}: Dictionary containing processed metrics from the evaluator.
        """
        try:
            # Initialize the structure
            aggregated_metrics = {}
            for evaluator_name in final_metric_dict.keys():
                aggregated_metrics[evaluator_name] = {
                    "metrics": {},
                    "aggregated_metric_label": {},
                    "metadata": {},
                    "metadata_label": {},
                }

            # Gather metrics
            for evaluator_name, metric_dict in final_metric_dict.items():
                # Organize metrics by label
                for key, value in metric_dict.items():
                    if key.startswith("metadata/"):
                        aggregated_metrics[evaluator_name]["metadata"][key] = value
                    elif key.startswith("metadata_label/"):
                        # These are per-label metrics, extract label name and organize
                        # Example: T4MetricV2/car_AP_center_distance_0.5
                        parts = key.split("/")[1].split("_")
                        label_name = parts[1]  # car, truck, etc.
                        if label_name not in aggregated_metrics[evaluator_name]["metadata_label"]:
                            aggregated_metrics[evaluator_name]["metadata_label"][label_name] = {}

                        aggregated_metrics[evaluator_name]["metadata_label"][label_name][key] = value
                    elif key.startswith("T4MetricV2/mAP_") or key.startswith("T4MetricV2/mAPH_"):
                        # These are overall metrics, put them in the metrics section
                        aggregated_metrics[evaluator_name]["metrics"][key] = value
                    else:
                        # These are per-label metrics, extract label name and organize
                        # Example: T4MetricV2/car_AP_center_distance_0.5
                        parts = key.split("/")[1].split("_")
                        label_name = parts[0]  # car, truck, etc.

                        if label_name not in aggregated_metrics[evaluator_name]["aggregated_metric_label"]:
                            aggregated_metrics[evaluator_name]["aggregated_metric_label"][label_name] = {}

                        aggregated_metrics[evaluator_name]["aggregated_metric_label"][label_name][key] = value

            # Write to JSON file
            output_path = self.output_dir / aggregated_metric_file_name
            with open(output_path, "w") as aggregated_file:
                json.dump(aggregated_metrics, aggregated_file, indent=4)

            self.logger.info(f"Aggregated metrics written to: {output_path}")
            return aggregated_metrics

        except Exception as e:
            self.logger.error(f"Failed to write aggregated metrics: {e}")
            raise

    def _parse_scene_id(self, lidar_path: str) -> str:
        """Parse scene ID from the LiDAR file path.

        Removes the `data_root` prefix and the trailing `/data` section.

        Args:
            lidar_path (str): The full file path of the LiDAR data.
            Example of the lidar_path: 'db_j6_v1/43e6e09a-93ce-488f-8f40-515187bc2753/2/data/LIDAR_CONCAT/0.pcd.bin'

        Returns:
            str: The extracted scene ID, or "unknown" if extraction fails.
            Example of the extracted scene ID: 'db_j6_v1/43e6e09a-93ce-488f-8f40-515187bc2753/2'
        """
        # TODO(vividf): This will be eventually moved to t4_devkit

        if not lidar_path or not lidar_path.startswith(self.data_root):
            return _UNKNOWN

        # Remove the data_root prefix
        relative_path = lidar_path[len(self.data_root) :].lstrip("/")  # Remove leading slash if exists
        path_parts = relative_path.split("/")

        # Extract scene ID before "data" section
        try:
            data_index = path_parts.index("data")
            return "/".join(path_parts[:data_index])
        except ValueError:
            return _UNKNOWN

    def _parse_ground_truth_from_sample(self, time: float, data_sample: Dict[str, Any]) -> FrameGroundTruth:
        """Parses ground truth objects from the given data sample.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            data_sample (Dict[str, Any]): A dictionary containing the ground truth data,
                                        including 3D bounding boxes, labels, and point counts.

        Returns:
            FrameGroundTruth: A structured representation of the ground truth objects,
                            including position, orientation, shape, velocity, and labels.
        """

        # Extract evaluation annotation info for the current sample
        eval_info: dict = data_sample.get("eval_ann_info", {})

        # Ensure sample_idx is int for internal identity
        sample_idx_raw = data_sample.get("sample_idx", None)
        if sample_idx_raw is None:
            raise ValueError("sample_idx is required in data_sample")
        sample_id_int = int(sample_idx_raw)  # Guarantee it's int
        # Convert to pure digit string for FrameGroundTruth.frame_name (library requires str)
        frame_name = str(sample_id_int)  # e.g., "123"

        # TODO(vividf): Double check this changes
        # location: str = data_sample.get("city", _UNKNOWN)
        # vehicle_type: str = data_sample.get("vehicle_type", _UNKNOWN)
        location: str = data_sample.get("city") or _UNKNOWN
        vehicle_type: str = data_sample.get("vehicle_type") or _UNKNOWN

        # gt_bboxes_3d: LiDARInstance3DBoxes with tensor of shape (N, 9)
        # Format per box: [x, y, z, l, w, h, yaw, vx, vy]
        gt_bboxes_3d: LiDARInstance3DBoxes = eval_info.get("gt_bboxes_3d", LiDARInstance3DBoxes([]))
        bboxes: np.ndarray = gt_bboxes_3d.tensor.cpu().numpy()

        # gt_labels_3d: (N,) array of class indices (e.g., [0, 1, 2, 3, ...])
        gt_labels_3d: np.ndarray = eval_info.get("gt_labels_3d", np.array([]))

        # num_lidar_pts: (N,) array of int, number of LiDAR points inside each GT box
        num_lidar_pts: np.ndarray = eval_info.get("num_lidar_pts", np.array([]))

        dynamic_objects = [
            bbox_to_dynamic_object(
                bbox=bbox,
                label=labels_index_to_label(self.class_names, int(label)),
                score=1.0,
                frame_id=self.main_evaluator_frame_id,
                unix_time=time,
                num_pts=int(num_pts),
            )
            for bbox, label, num_pts in zip(bboxes, gt_labels_3d, num_lidar_pts)
            if not (np.isnan(label) or np.isnan(num_pts) or np.any(np.isnan(bbox)))
        ]

        return FrameGroundTruth(
            unix_time=time,
            frame_name=frame_name,  # Pure digit string (e.g., "123") as required by library
            objects=dynamic_objects,
            transforms=None,
            raw_data=None,
            frame_prefix=location + "/" + vehicle_type,
        )

    def _parse_predictions_from_sample(
        self, time: float, data_sample: Dict[str, Any], ground_truth_objects: FrameGroundTruth
    ) -> PerceptionFrame:
        """
        Parses predicted objects from the data sample and creates a perception frame result.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            data_sample (Dict[str, Any]): A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.
            ground_truth_objects (FrameGroundTruth): The ground truth data corresponding to the current frame.

        Returns:
            PerceptionFrame: A structured result containing the predicted objects and ground truth objects.
        """
        pred_3d: Dict[str, Any] = data_sample.get("pred_instances_3d", {})

        # bboxes_3d: LiDARInstance3DBoxes with tensor of shape (N, 9)
        # Format per box: [x, y, z, l, w, h, yaw, vx, vy]
        bboxes_3d = pred_3d.get("bboxes_3d", LiDARInstance3DBoxes([]))
        bboxes: np.ndarray = bboxes_3d.tensor.cpu().numpy()

        # scores_3d: (N,) Tensor of detection confidence scores
        scores: torch.Tensor = pred_3d.get("scores_3d", torch.empty(0)).cpu()
        # labels_3d: (N,) Tensor of predicted class indices
        labels: torch.Tensor = pred_3d.get("labels_3d", torch.empty(0)).cpu()
        estimated_objects = [
            bbox_to_dynamic_object(
                bbox=bbox,
                label=labels_index_to_label(self.class_names, int(label)),
                score=float(score),
                frame_id=self.main_evaluator_frame_id,
                unix_time=time,
            )
            for bbox, score, label in zip(bboxes, scores, labels)
            if not (np.isnan(score) or np.isnan(label) or np.any(np.isnan(bbox)))
        ]

        return PerceptionFrame(
            unix_time=time,
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
        )

    def _save_perception_frame(self, scene_id: str, sample_idx: int, perception_frame: PerceptionFrame) -> None:
        """
        Stores the processed perception result in self.results following the format:
        [
            {
                <scene_id>:
                    {<sample_idx>: <PerceptionFrame>},
                    {<sample_idx>: <PerceptionFrame>},
            },
            {
                <scene_id>:
                    {<sample_idx>: <PerceptionFrame>},
                    {<sample_idx>: <PerceptionFrame>},
            },
        ]

        Args:
            scene_id (str): The identifier for the scene to which the result belongs.
            sample_idx (int): The index of the sample within the scene. Must be int type.
            perception_frame (PerceptionFrame): The processed perception result for the given sample.
        """
        # Ensure sample_idx is int type (matching FrameInput.sample_id type)
        assert isinstance(sample_idx, int), f"sample_idx must be int, got {type(sample_idx)}: {sample_idx}"

        # Use sample_idx (int) directly as dict key (matching FrameInput.sample_id type)
        index = self.scene_id_to_index_map.get(scene_id, None)
        if index is not None:
            self.results[index][scene_id][sample_idx] = perception_frame
        else:
            # New scene: append to results and record its index
            self.results.append({scene_id: {sample_idx: perception_frame}})
            self.scene_id_to_index_map[scene_id] = len(self.results) - 1

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.manager import PerceptionEvaluationManager
from pyquaternion import Quaternion



from collections import defaultdict

# Test
from mmengine import load
import os
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmengine import load
from mmengine.logging import MMLogger, print_log
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox

from autoware_ml.detection3d.evaluation.t4metric.evaluation import T4DetectionConfig, T4DetectionEvaluation
from autoware_ml.detection3d.evaluation.t4metric.loading import t4metric_load_gt, t4metric_load_prediction



__all__ = ["T4MetricV2"]
_UNKNOWN = "unknown"


@METRICS.register_module()
class T4MetricV2(BaseMetric):
    """T4 format evaluation metric V2.
    Args:
        data_root (str):
            Path of dataset root.
        ann_file (str):
            Path of annotation file.
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
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
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
        super(T4MetricV2, self).__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.data_root = data_root

        self.class_names = class_names
        self.name_mapping = name_mapping

        if name_mapping is not None:
            self.class_names = [self.name_mapping.get(name, name) for name in self.class_names]

        self.results_pickle_path: Optional[Path] = Path(results_pickle_path) if results_pickle_path else None
        if self.results_pickle_path is not None and self.results_pickle_path.suffix != ".pkl":
            raise ValueError(f"results_pickle_path must end with '.pkl', got: {self.results_pickle_path}")

        self.results_pickle_exists = True if self.results_pickle_path and self.results_pickle_path.exists() else False

        self.target_labels = [AutowareLabel[label.upper()] for label in self.class_names]

        self.perception_evaluator_configs = PerceptionEvaluationConfig(**perception_evaluator_configs)

        self.critical_object_filter_config = CriticalObjectFilterConfig(
            evaluator_config=self.perception_evaluator_configs, **critical_object_filter_config
        )
        self.frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=self.perception_evaluator_configs, **frame_pass_fail_config
        )

        self.metrics_config = MetricsScoreConfig(
            self.perception_evaluator_configs.evaluation_task, target_labels=self.target_labels
        )

        self.scene_id_to_index_map: Dict[str, int] = {}  # scene_id to index map in self.results

        self.logger = MMLogger.get_current_instance()


        # testing
        # load annotations
        self.backend_args = None
        self.data_infos = load(self.ann_file, backend_args=self.backend_args)["data_list"]
        scene_tokens, directory_names = self._get_scene_info(self.data_infos)
        self.loaded_scenes, self.scene_tokens, self.directory_names = self._load_scenes(
            self.data_root,
            scene_tokens,
            directory_names,
        )

        eval_class_range = {
            "car": 121,
            "truck": 121,
            "bus": 121,
            "bicycle": 121,
            "pedestrian": 121,
        }

        # Load eval detection config
        eval_config_dict = {
            "class_names": tuple(self.class_names),
            "class_range": eval_class_range,
            "dist_fcn": "center_distance",
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5,
        }
        self.eval_detection_configs = T4DetectionConfig.deserialize(eval_config_dict)

        # dict {sample_token: ... , sample_token: ...}
        self.samples_gts = self.helper() 

        print("get samples_gts")


    # override of BaseMetric.process
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model and the ground truth of dataset  I am
        """

        if self.results_pickle_exists:
            # Skip processing if result pickle already exists
            return

        #print("data_samples: ", data_samples)
        for data_sample in data_samples:

            # print("data_sample: ", data_sample)
            current_time = data_sample["timestamp"]
            scene_id = self.parse_scene_id(data_sample["lidar_path"])
            ego2global, lidar2ego = self.get_transformation_matrices(data_sample)

            #print("Ego to Global Matrix:\n", ego2global)
            #print("Lidar to Ego Matrix:\n", lidar2ego)

            #print("lidar2ego: ", lidar2ego)
            #print("ego2global: ", ego2global)
            # Optional: if you want lidar2global
            lidar2global = ego2global @ lidar2ego
            
            #print("Lidar to Global Matrix:\n", lidar2global)
            #frame_ground_truth = self.parse_ground_truth_from_sample(current_time, data_sample, lidar2global)
            #print("get from sample")
            frame_ground_truth = self.parse_ground_truth_from_t4dataset(current_time, data_sample, lidar2global)
            #print("get from dataset")
            perception_frame_result = self.parse_predictions_from_sample(current_time, data_sample, frame_ground_truth, lidar2global)
            self.save_perception_results(scene_id, data_sample["sample_idx"], perception_frame_result)

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

        if self.results_pickle_exists:
            results = self.load_results_from_pickle(self.results_pickle_path)
        elif self.results_pickle_path:
            self.save_results_to_pickle(self.results_pickle_path)

        evaluator = PerceptionEvaluationManager(evaluation_config=self.perception_evaluator_configs)

        scenes, scene_metrics = self.init_scene_metrics_from_results(results)

        for scene_id, samples in scenes.items():
            for sample_id, frame_result in samples.items():
                object_with_results = frame_result.object_results
                estimated_objects = [obj.estimated_object for obj in object_with_results]

                frame_result: PerceptionFrameResult = evaluator.add_frame_result(
                    unix_time=time.time(),
                    ground_truth_now_frame=frame_result.frame_ground_truth,
                    estimated_objects=estimated_objects,
                    critical_object_filter_config=self.critical_object_filter_config,
                    frame_pass_fail_config=self.frame_pass_fail_config,
                )

                                
                # frame_result: PerceptionFrameResult = evaluator.add_frame_result_vivid(
                #     unix_time=time.time(),
                #     ground_truth_now_frame=frame_result.frame_ground_truth,
                #     estimated_objects=estimated_objects,
                #     critical_object_filter_config=self.critical_object_filter_config,
                #     # frame_pass_fail_config=self.frame_pass_fail_config,
                # )

                for map_instance in frame_result.metrics_score.maps:
                    self.process_map_instance(map_instance, scene_metrics[scene_id][sample_id])

        # final_metric_score = evaluator.get_scene_result_vivid()
        final_metric_score = evaluator.get_scene_result()
        self.logger.info(f"final metrics result {final_metric_score}")

        metric_dict = {}
        aggregated_metrics = {"aggregated_metrics": {}}

        # Iterate over the list of maps in final_metric_score
        for map_instance in final_metric_score.maps:
            self.process_map_instance(map_instance, metric_dict)

            for key, value in metric_dict.items():
                # Extract label name from metric_dict's keys
                # Example: T4MetricV2/car_AP_center_distance_bev_0.5
                label = key.split("/")[1].split("_")[0]
                aggregated_metrics["aggregated_metrics"].setdefault(label, {})
                aggregated_metrics["aggregated_metrics"][label][key] = value

        with open("scene_metrics.json", "w") as scene_file:
            json.dump(scene_metrics, scene_file, indent=4)

        with open("aggregated_metrics.json", "w") as agg_file:
            json.dump(aggregated_metrics, agg_file, indent=4)

        return metric_dict

    def convert_index_to_label(self, bbox_label_index: int) -> Label:
        """
        Convert a bounding box label index into a Label object containing the corresponding AutowareLabel.

        Args:
            bbox_label_index (int): Index from the model output representing the predicted class.

        Returns:
            Label: A Label object with the corresponding AutowareLabel enum and class name string.
        """
        class_name = self.class_names[bbox_label_index] if 0 <= bbox_label_index < len(self.class_names) else _UNKNOWN
        autoware_label = AutowareLabel.__members__.get(class_name.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=class_name)

    def parse_scene_id(self, lidar_path: str) -> str:
        """parse scene ID from the LiDAR file path.

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

    def parse_ground_truth_from_sample(self, time: float, data_sample: Dict[str, Any], lidar2global: np.ndarray) -> FrameGroundTruth:
        """Parses ground truth objects from the given data sample.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.

        Returns:
            FrameGroundTruth: A structured representation of the ground truth objects,
                            including position, orientation, shape, velocity, and labels.
        """

        # Extract evaluation annotation info for the current sample
        eval_info: dict = data_sample.get("eval_ann_info", {})
        sample_id: str = data_sample.get("sample_idx", _UNKNOWN)

        # gt_bboxes_3d: LiDARInstance3DBoxes with tensor of shape (N, 9)
        # Format per box: [x, y, z, l, w, h, yaw, vx, vy]
        gt_bboxes_3d: LiDARInstance3DBoxes = eval_info.get("gt_bboxes_3d", LiDARInstance3DBoxes([]))
        bboxes: np.ndarray = gt_bboxes_3d.tensor.cpu().numpy()

        # gt_labels_3d: (N,) array of class indices (e.g., [0, 1, 2, 3, ...])
        gt_labels_3d: np.ndarray = eval_info.get("gt_labels_3d", np.array([]))

        # num_lidar_pts: (N,) array of int, number of LiDAR points inside each GT box
        num_lidar_pts: np.ndarray = eval_info.get("num_lidar_pts", np.array([]))

        # target = np.array([-4.7819214, -35.89899])

        # for box in bboxes:
        #     if np.allclose(box[:2], target, atol=1e-2):  # 可調整容許誤差
        #         print("bbbbbbb: ", box)

        dynamic_objects = []
        for bbox, label, num_pts in zip(bboxes, gt_labels_3d, num_lidar_pts):
            if np.isnan(label) or np.isnan(num_pts) or np.any(np.isnan(bbox)):
                continue

            radius = np.linalg.norm(bbox[:2], 2)
            if radius > 121:
                continue

            if np.allclose(bbox[:3], [-60.19682, -37.06706, 0.8657075], atol=1e-3):
                test_point = np.array([65453.005956938156, 642.783003459129, 716.7769521839763, 1.0])
                test_box = np.linalg.inv(lidar2global) @ test_point
                # print("Inverse transformed test box:", test_box)

            # change from center to bottom
            # print("before origin change: ", bbox[:3])
            bbox[2] += bbox[5] * 0.5

            # print("after origin change, before tranform ground truth: ", bbox[:3])
            # print("lidar2global: ", lidar2global)
            # === 1. 轉換 position 到 global ===
            pos_homogeneous = np.append(bbox[:3], 1.0)  # (x, y, z, 1)
            global_position = (lidar2global @ pos_homogeneous)[:3]  # 取前 3 維

            # === 2. 將 yaw 轉為 quaternion，並轉換到 global frame ===
            q_yaw = Quaternion(axis=[0, 0, 1], angle=bbox[6])  # LiDAR frame 下的 yaw
            q_rot = Quaternion(matrix=lidar2global[:3, :3], rtol=1e-5, atol=1e-7)  # global 的旋轉
            global_orientation = q_rot * q_yaw  # 將 yaw 套入 global frame
            
            print("global_position: ", global_position)
            # print("after tranfroms ground truth: ", global_position)
            dynamic_objects.append(
                DynamicObject(
                    unix_time=time,
                    frame_id=self.perception_evaluator_configs.frame_id,
                    position=tuple(global_position),
                    orientation=global_orientation,  # still in LiDAR frame
                    shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(bbox[3:6])),
                    velocity=(bbox[7], bbox[8], 0.0),
                    semantic_score=1.0,
                    semantic_label=self.convert_index_to_label(int(label)),
                    pointcloud_num=int(num_pts),
                )
            ) 
        return FrameGroundTruth(
            unix_time=time,
            frame_name=sample_id,
            objects=dynamic_objects,
            transforms=None,
            raw_data=None,
        )
    

    def parse_ground_truth_from_t4dataset(self, time: float, data_sample: Dict[str, Any], lidar2global: np.ndarray) -> FrameGroundTruth:
        """Parses ground truth objects from the given data sample.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.

        Returns:
            FrameGroundTruth: A structured representation of the ground truth objects,
                            including position, orientation, shape, velocity, and labels.
        """

        # Extract evaluation annotation info for the current sample
        # eval_info: dict = data_sample.get("eval_ann_info", {})
        sample_id: str = data_sample.get("sample_idx", _UNKNOWN)
        


        dynamic_objects = []

        #print("sample_id: ", sample_id)


        sample_token = self.data_infos[sample_id]["token"]
        #print("sample_token: ", sample_token)
        sample_gts = self.samples_gts[sample_token]
        for gt in sample_gts:
            #print("gt.translation: ", gt.translation)
            dynamic_objects.append(
                DynamicObject(
                    unix_time=time,
                    frame_id=self.perception_evaluator_configs.frame_id,
                    position=gt.translation,
                    orientation=gt.rotation,  # still in LiDAR frame
                    shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(gt.size)),
                    velocity=(gt.velocity, 0.0),
                    semantic_score=1.0,
                    semantic_label=self.convert_name_to_label(gt.detection_name),
                    pointcloud_num=int(gt.num_pts),
                )
            ) 
        
        # print("generate gt")

        return FrameGroundTruth(
            unix_time=time,
            frame_name=sample_id,
            objects=dynamic_objects,
            transforms=None,
            raw_data=None,
        )
    

    # Tests
    def convert_name_to_label(self, bbox_label_str: str) -> Label:
        """
        Convert a bounding box label index into a Label object containing the corresponding AutowareLabel.

        Args:
            bbox_label_index (int): Index from the model output representing the predicted class.

        Returns:
            Label: A Label object with the corresponding AutowareLabel enum and class name string.
        """
        autoware_label = AutowareLabel.__members__.get(bbox_label_str.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=bbox_label_str)

    def parse_predictions_from_sample(
        self, time: float, data_sample: Dict[str, Any], frame_ground_truth: FrameGroundTruth, lidar2global: np.ndarray
    ) -> PerceptionFrameResult:
        """
        Parses predicted objects from the data sample and creates a perception frame result.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            data_sample (Dict[str, Any]): A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.
            frame_ground_truth (FrameGroundTruth): The ground truth data corresponding to the current frame.

        Returns:
            PerceptionFrameResult: A structured result containing the predicted objects, frame ground truth, and evaluation configurations.
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

        dynamic_objects_with_perception = []

        for bbox, score, label in zip(bboxes, scores, labels):
            if np.isnan(score) or np.isnan(label) or np.any(np.isnan(bbox)):
                continue

            radius = np.linalg.norm(bbox[:2], 2)
            if radius > 121:
                continue


            # change from center to bottom
            # print("before origin change: ", bbox[:3])
            bbox[2] += bbox[5] * 0.5

            # print("after origin change, before tranform prediction: ", bbox[:3])
            # === 1. 轉換 position 到 global ===
            pos_homogeneous = np.append(bbox[:3], 1.0)  # (x, y, z, 1)
            global_position = (lidar2global @ pos_homogeneous)[:3]  # 取前 3 維

            #print("lidar2global: ", lidar2global)
            # === 2. 將 yaw 轉為 quaternion，並轉換到 global frame ===
            q_yaw = Quaternion(axis=[0, 0, 1], angle=bbox[6])  # LiDAR frame 下的 yaw
            q_rot = Quaternion(matrix=lidar2global[:3, :3], rtol=1e-5, atol=1e-7)  # global 的旋轉
            global_orientation = q_rot * q_yaw  # 將 yaw 套入 global frame

            # print("after tranform prediction: ", global_position)
            # === 3. 包成 DynamicObject ===
            dynamic_object = DynamicObject(
                unix_time=time,
                frame_id=self.perception_evaluator_configs.frame_id,
                position=tuple(global_position),
                orientation=global_orientation,
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(bbox[3:6])),
                velocity=(bbox[7], bbox[8], 0.0),
                semantic_score=float(score),
                semantic_label=self.convert_index_to_label(int(label)),
            )

            # === 4. 包進 Result 結構 ===
            dynamic_objects_with_perception.append(
                DynamicObjectWithPerceptionResult(
                    estimated_object=dynamic_object,
                    ground_truth_object=None,
                )
            )


        return PerceptionFrameResult(
            object_results=dynamic_objects_with_perception,
            frame_ground_truth=frame_ground_truth,
            metrics_config=self.metrics_config,
            critical_object_filter_config=self.critical_object_filter_config,
            frame_pass_fail_config=self.frame_pass_fail_config,
            unix_time=time,
            target_labels=self.target_labels,
        )
    
    def get_transformation_matrices(
        self, data_sample: Dict[str, Any], sensor_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve both ego2global and lidar2ego transformation matrices from the data_sample.

        Args:
            data_sample (Dict[str, Any]): A dictionary containing ego2global and lidar2cam info.
            sensor_idx (int): Index to select a specific lidar2cam transform (default: 0).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (ego2global, lidar2ego) as 4x4 numpy arrays.
        """
        ego2global = np.eye(4)
        lidar2ego = np.eye(4)

        if "ego2global" in data_sample:
            ego2global = np.array(data_sample["ego2global"])
        else:
            self.logger.warning("ego2global not found in data_sample")

        if "lidar_points" in data_sample:
            lidar2ego = np.array(data_sample["lidar_points"]["lidar2ego"])
        else:
            self.logger.warning("lidar2ego not found in data_sample")

        return ego2global, lidar2ego

    
    

    def save_perception_results(
        self, scene_id: str, sample_idx: int, perception_frame_result: PerceptionFrameResult
    ) -> None:
        """
        Stores the processed perceptoin result in self.results following the format.
        [
            {
                <scence_id>:
                    {<sample_idx>: <PerceptionFrameResult>},
                    {<sample_idx>: <PerceptionFrameResult>},

            },
            {
                <scence_id>:
                    {<sample_idx>: <PerceptionFrameResult>},
                    {<sample_idx>: <PerceptionFrameResult>},

            },
        ]

        Args:
            scene_id (str): The identifier for the scene to which the result belongs.
            sample_idx (int): The index of the sample within the scene.
            perception_frame_result (PerceptionFrameResult): The processed perception result for the given sample.
        """

        index = self.scene_id_to_index_map.get(scene_id, None)
        if index is not None:
            self.results[index][scene_id][sample_idx] = perception_frame_result
        else:
            # New scene: append to results and record its index
            self.results.append({scene_id: {sample_idx: perception_frame_result}})
            self.scene_id_to_index_map[scene_id] = len(self.results) - 1

    def save_results_to_pickle(self, path: Path) -> None:
        """Save self.results to the given pickle file path.

        Args:
            path (Path): The full path where the pickle file will be saved.
        """
        self.logger.info(f"Saving predictions and ground truth result to pickle: {path.resolve()}")

        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.results, f)

    def load_results_from_pickle(self, path: Path) -> List[Dict]:
        """Load results from a pickle file.

        Args:
            path (Path): The full path to the pickle file.

        Returns:
            List[Dict]: The deserialized results from the pickle file.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        self.logger.info(f"Loading pickle from: {path.resolve()}")
        with open(path, "rb") as f:
            results = pickle.load(f)

        return results

    def process_map_instance(self, map_instance, metrics_store):
        matching_mode = map_instance.matching_mode.value  # e.g., "Center Distance" or "Plane Distance"

        for ap in map_instance.aps:
            # Each Ap instance is configured for a single label and threshold,
            # so we extract the only elements from target_labels and matching_threshold_list.
            label = ap.target_labels[0].value  # AutowareLabel
            threshold = ap.matching_threshold_list[0]
            ap_value = ap.ap

            # Construct the metric key
            key = f"T4MetricV2/{label}_AP_{matching_mode.lower().replace(' ', '_')}_{threshold}"
            metrics_store[key] = ap_value

    def init_scene_metrics_from_results(self, results: list[Dict[str, Dict[str, Any]]]) -> tuple[dict, dict]:
        """
        Flattens scene dictionaries from the results and initializes scene_metrics structure.

        Args:
            results (list): List of dictionaries mapping scene_id to sample_id-frame_result pairs.

        Returns:
            tuple:
                - scenes (dict): Flattened dict of {scene_id: {sample_id: frame_result}}.
                - scene_metrics (dict): Initialized dict of {scene_id: {sample_id: dict}} for metric storage.
        """
        scenes = {scene_id: samples for scene in results for scene_id, samples in scene.items()}

        scene_metrics = {
            scene_id: {sample_id: {} for sample_id in samples.keys()} for scene_id, samples in scenes.items()
        }

        return scenes, scene_metrics



    ### Testing, remove this later


    def helper(self):
        all_gt_boxes = defaultdict(list)

        for scene in self.scene_tokens:
            gts = self.get_gt_bboxes(scene)
            print("############################")
            print("gts: ", gts)

            for key, boxes in gts.boxes.items():
                all_gt_boxes[key].extend(boxes)

        return all_gt_boxes

    def get_gt_bboxes(
            self,
            scene_token: str,
        ) -> EvalBoxes:
        """Evaluate the results in T4 format.

        Args:
            scene_token (str): The scene token.
        Returns:
            Dict[str, float]: The evaluation results.
        """

        nusc = self.loaded_scenes[scene_token]

        filter_attributes = [
            ("vehicle.bicycle", "vehicle_state.parked"),
            ("vehicle.bicycle", "cycle_state.without_rider"),
            ("vehicle.bicycle", "motorcycle_state.without_rider"),
            ("vehicle.motorcycle", "vehicle_state.parked"),
            ("vehicle.motorcycle", "cycle_state.without_rider"),
            ("vehicle.motorcycle", "motorcycle_state.without_rider"),
            ("bicycle", "vehicle_state.parked"),
            ("bicycle", "cycle_state.without_rider"),
            ("bicycle", "motorcycle_state.without_rider"),
            ("motorcycle", "vehicle_state.parked"),
            ("motorcycle", "cycle_state.without_rider"),
            ("motorcycle", "motorcycle_state.without_rider"),
        ]

        gt_boxes = t4metric_load_gt(
            nusc,
            self.eval_detection_configs,
            scene_token,
            post_mapping_dict=self.name_mapping,
            filter_attributions=filter_attributes,
            verbose=True
        )
    
        return gt_boxes

    @staticmethod
    def _get_scene_info(data_infos: List[dict]) -> Tuple[List[str], List[str]]:
        """Get scene tokens and directory names from data infos.

        Args:
            data_infos (List[dict]): The data infos.

        Returns:
            List[str]: The scene tokens.
            List[str]: The directory names.
        """
        scene_tokens = []
        directories = []
        for info in data_infos:
            scene_token = info["scene_token"]
            # ['db_jpntaxi_v1', '3a13032b-6045-4db4-8632-9c52c3dd2fd9', '0', 'data', 'LIDAR_CONCAT', '98.pcd.bin']
            directory_list = info["lidar_points"]["lidar_path"].split("/")
            # 'db_jpntaxi_v1/3a13032b-6045-4db4-8632-9c52c3dd2fd9/0'
            directory = osp.join(*directory_list[0:3])
            if directory not in directories:
                scene_tokens.append(scene_token)
                directories.append(directory)
        return scene_tokens, directories
    


    @staticmethod
    def _load_scenes(
        data_root: str,
        scene_tokens: List[str],
        directory_names: List[str],
    ) -> Tuple[Dict[str, NuScenes], List[str], List[str]]:
        """Load scenes from data infos.

        Returns:
            Dict[str, NuScenes]: The loaded scenes.
        """

        loaded_dirs = {}
        existing_dirs = []
        existing_scenes = []
        for directory, scene_token in zip(directory_names, scene_tokens):
            scene_directory_path = os.path.join(data_root, directory)
            if os.path.exists(scene_directory_path):
                loaded_dirs[scene_token] = NuScenes(
                    version="annotation",
                    dataroot=scene_directory_path,
                    verbose=False,
                )
                existing_dirs.append(directory)
                existing_scenes.append(scene_token)
            else:
                print(f"Skipped non-existing {scene_directory_path} in T4Metric")
        return loaded_dirs, existing_scenes, existing_dirs
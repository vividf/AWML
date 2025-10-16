"""
Detection metrics for deployment evaluation.

Provides functions to compute AP, mAP, and other detection metrics.
"""

from typing import Dict, List, Tuple

import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        box1: [x, y, w, h] format
        box2: [x, y, w, h] format

    Returns:
        IoU value
    """
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    class_id: int = None,
) -> float:
    """
    Compute Average Precision for a single class.

    Args:
        predictions: List of prediction dicts with 'bbox', 'label', 'score'
        ground_truths: List of ground truth dicts with 'bbox', 'label'
        iou_threshold: IoU threshold for matching
        class_id: Class ID to compute AP for (None for all classes)

    Returns:
        Average Precision value
    """
    # Filter by class
    if class_id is not None:
        predictions = [p for p in predictions if p["label"] == class_id]
        ground_truths = [g for g in ground_truths if g["label"] == class_id]

    if len(ground_truths) == 0:
        # If no ground truths exist for this class, AP is undefined
        # Return 0.0 regardless of predictions to avoid false positives
        return 0.0

    if len(predictions) == 0:
        return 0.0

    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # Mark ground truths as not matched
    gt_matched = [False] * len(ground_truths)

    tp = []
    fp = []

    for pred in predictions:
        pred_box = pred["bbox"]

        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue

            gt_box = gt["bbox"]
            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if it's a match
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP (11-point interpolation)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return ap


def compute_map_coco(
    predictions_list: List[List[Dict]],
    ground_truths_list: List[List[Dict]],
    num_classes: int,
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """
    Compute COCO-style mAP metrics.

    Args:
        predictions_list: List of predictions for each image
        ground_truths_list: List of ground truths for each image
        num_classes: Number of classes
        iou_thresholds: IoU thresholds (default: [0.5:0.95:0.05])

    Returns:
        Dictionary with mAP metrics:
        - mAP: mean AP across all IoU thresholds
        - mAP_50: AP at IoU=0.5
        - mAP_75: AP at IoU=0.75
        - per_class_ap: AP for each class
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)

    # Flatten predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    for preds, gts in zip(predictions_list, ground_truths_list):
        all_predictions.extend(preds)
        all_ground_truths.extend(gts)

    # Compute AP for each class and IoU threshold
    aps = []
    per_class_ap = {}

    for class_id in range(num_classes):
        class_aps = []
        for iou_thr in iou_thresholds:
            ap = compute_ap(all_predictions, all_ground_truths, iou_threshold=iou_thr, class_id=class_id)
            class_aps.append(ap)

        # Average across IoU thresholds
        mean_ap = np.mean(class_aps)
        per_class_ap[class_id] = mean_ap
        aps.append(class_aps)

    aps = np.array(aps)

    # Compute overall metrics
    results = {
        "mAP": float(np.mean(aps)),
        "mAP_50": float(np.mean(aps[:, 0])) if len(iou_thresholds) > 0 else 0.0,
        "mAP_75": float(np.mean(aps[:, 5])) if len(iou_thresholds) > 5 else 0.0,
        "per_class_ap": per_class_ap,
    }

    return results


def compute_3d_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 3D IoU between two 3D bounding boxes.

    Args:
        box1: [x, y, z, w, l, h, yaw] format
        box2: [x, y, z, w, l, h, yaw] format

    Returns:
        3D IoU value

    Note: This is a simplified version. For production, use mmdet3d's implementation.
    """
    # This is a placeholder - proper 3D IoU is more complex
    # For production, should use mmdet3d.core.bbox.iou_calculators

    # Simple approximation: use BEV IoU
    # Convert to BEV boxes [x, y, w, l, yaw]
    bev1 = box1[[0, 1, 3, 4, 6]]
    bev2 = box2[[0, 1, 3, 4, 6]]

    # For now, return a simple overlap measure
    # TODO: Implement proper 3D IoU or use mmdet3d's implementation
    return 0.0


def compute_3d_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.7,
    class_id: int = None,
) -> float:
    """
    Compute Average Precision for 3D detection.

    Args:
        predictions: List of prediction dicts with 'bbox_3d', 'label', 'score'
        ground_truths: List of ground truth dicts with 'bbox_3d', 'label'
        iou_threshold: 3D IoU threshold for matching
        class_id: Class ID to compute AP for

    Returns:
        Average Precision value

    Note: For production, should use mmdet3d's evaluation metrics.
    """
    # This is a simplified version
    # For production, should use mmdet3d.core.evaluation

    # Filter by class
    if class_id is not None:
        predictions = [p for p in predictions if p["label"] == class_id]
        ground_truths = [g for g in ground_truths if g["label"] == class_id]

    if len(ground_truths) == 0:
        # If no ground truths exist for this class, AP is undefined
        # Return 0.0 regardless of predictions to avoid false positives
        return 0.0

    if len(predictions) == 0:
        return 0.0

    # For now, return a placeholder
    # TODO: Implement proper 3D AP or integrate with mmdet3d
    return 0.0

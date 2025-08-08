import os

import numpy as np
import torch
import torch.nn.functional as F
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmengine.registry import DATASETS, TRANSFORMS

# Use the correct deploy and model config files that exist in the project
deploy_cfg = "projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_onnxruntime.py"
model_cfg = "projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py"
device = "cpu"
backend_model = ["/workspace/work_dirs/end2end_train_200_ptq.quant.onnx"]
# backend_model = ['/workspace/work_dirs/end2end.onnx']

# Check if config files exist
if not os.path.exists(deploy_cfg):
    raise FileNotFoundError(f"Deploy config file not found: {deploy_cfg}")
if not os.path.exists(model_cfg):
    raise FileNotFoundError(f"Model config file not found: {model_cfg}")

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)

# Check if ONNX model exists
if not os.path.exists(backend_model[0]):
    print(f"Warning: ONNX model not found at {backend_model[0]}")
    print("You need to export the model to ONNX first using the deploy script.")
    print(
        "Example: python projects/CalibrationStatusClassification/deploy/main.py --deploy-cfg projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch_onnxruntime.py --model-cfg projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py --checkpoint your_checkpoint.pth"
    )
    exit(1)

model = task_processor.build_backend_model(backend_model)

print("Testing ONNX model with entire test dataset...")

# Build dataset directly from config
dataset_cfg = model_cfg.test_dataloader.dataset
dataset = DATASETS.build(dataset_cfg)

print(f"Test dataset created with {len(dataset)} samples")

# Lists to store predictions and ground truth
all_predictions = []
all_ground_truth = []
all_probabilities = []

# Evaluate entire dataset
for sample_idx in range(len(dataset)):
    if sample_idx % 10 == 0:
        print(f"Processing sample {sample_idx + 1}/{len(dataset)}")

    # Get a single sample from dataset
    data_sample = dataset[sample_idx]

    # Convert to batch format expected by model
    data_batch = {"inputs": data_sample["inputs"].unsqueeze(0)}  # Add batch dimension

    # do model inference
    with torch.no_grad():
        result = model.test_step(data_batch)

    # Extract ground truth label
    gt_label = data_sample["data_samples"].gt_label.item()

    # Process result
    if isinstance(result, (list, tuple)):
        res = result[0]  # Take first result
    else:
        res = result

    if hasattr(res, "pred_score"):
        # Apply softmax to convert logits to probabilities
        logits = res.pred_score
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities.max().item()

        # Store results
        all_predictions.append(predicted_class)
        all_ground_truth.append(gt_label)
        all_probabilities.append(probabilities.cpu().numpy())

        # Print first few samples for debugging
        if sample_idx < 3:
            print(f"Sample {sample_idx + 1}: GT={gt_label}, Pred={predicted_class}, Confidence={confidence:.4f}")
            print(f"  Logits: {logits}")
            print(f"  Probabilities: {probabilities}")
    else:
        print(f"Warning: Sample {sample_idx} has no pred_score attribute")

# Calculate accuracy
all_predictions = np.array(all_predictions)
all_ground_truth = np.array(all_ground_truth)
all_probabilities = np.array(all_probabilities)

correct_predictions = (all_predictions == all_ground_truth).sum()
total_samples = len(all_predictions)
accuracy = correct_predictions / total_samples

print(f"\n{'='*50}")
print(f"ONNX Model Evaluation Results")
print(f"{'='*50}")
print(f"Total samples: {total_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calculate per-class accuracy
unique_classes = np.unique(all_ground_truth)
print(f"\nPer-class accuracy:")
for cls in unique_classes:
    cls_mask = all_ground_truth == cls
    cls_correct = (all_predictions[cls_mask] == all_ground_truth[cls_mask]).sum()
    cls_total = cls_mask.sum()
    cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0
    print(f"  Class {cls}: {cls_correct}/{cls_total} = {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")

# Calculate average confidence
avg_confidence = np.mean([prob.max() for prob in all_probabilities])
print(f"\nAverage confidence: {avg_confidence:.4f}")

# Show confusion matrix
print(f"\nConfusion Matrix:")
print(f"Predicted ->")
print(f"Actual    0    1")
for true_cls in unique_classes:
    row = []
    for pred_cls in unique_classes:
        count = ((all_ground_truth == true_cls) & (all_predictions == pred_cls)).sum()
        row.append(f"{count:4d}")
    print(f"  {true_cls}    {' '.join(row)}")

print(f"\nONNX model evaluation completed successfully!")
print(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

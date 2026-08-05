# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-batch overfit probe (AWML side of a cross-framework A/B).

Counterpart of ``python -m autoware_ml.tools.overfit_probe`` in autoware-ml.
Both write the same JSONL trace format so the traces can be compared step by
step with ``autoware_ml.tools.compare_overfit``.

Randomness is disabled by default so the trace is reproducible and comparable:
augmentation (GlobalRotScaleTransImage dropped, ResizeCropFlipRotImage pinned
to test mode), camera-order shuffle, GridMask, denoising queries, LR schedule
(constant LR, no warmup/cosine) and temporal memory carry-over.

Note on the learning rate: the reference recipe reaches its effective LR
through ``auto_scale_lr`` (5e-5 x 2 for a 16-sample global batch). This probe
does not scale anything, so pass the SAME ``--lr`` here and in the autoware-ml
probe (the reference recipe's effective value is 1e-4).

Usage:
    python tools/detection3d/overfit_probe.py \\
        projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore.py \\
        --weights pretrained/nuscenes_vov99_baseline_320x800.pth \\
        --steps 200 --batch-size 2 --lr 1e-4 \\
        --output parity_out/awml_overfit.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import DATASETS, MODELS, init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner import set_random_seed
from mmengine.utils import import_modules_from_strings

logger = logging.getLogger(__name__)

_AUGMENTATION_TYPES = ("GlobalRotScaleTransImage", "RandomFlip3D")


def parse_args() -> argparse.Namespace:
    """Parse probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Training config file")
    parser.add_argument("--weights", required=True, help="Checkpoint used as the starting point")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", required=True, help="JSONL trace output path")
    parser.add_argument(
        "--ann-file",
        default=None,
        help=(
            "Override the training ann_file. Pointing both probes at the small val pkl "
            "cuts dataset-build time from minutes to seconds while iterating."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument(
        "--lr", type=float, default=None, help="Constant LR (pass the same value in both probes)"
    )
    parser.add_argument("--keep-augmentation", action="store_true")
    parser.add_argument("--keep-grid-mask", action="store_true")
    parser.add_argument("--keep-dn", action="store_true")
    parser.add_argument("--keep-memory", action="store_true")
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Allow nondeterministic GPU kernels (faster, but two identical runs "
        "then differ by ~8%% on tail-mean loss, so traces are not comparable)",
    )
    return parser.parse_args()


def enable_determinism() -> None:
    """Make the run bitwise reproducible so two traces can be compared directly.

    Seeding alone is not enough: it pins step 0 exactly but nondeterministic GPU
    kernels let the 200-step trajectories diverge chaotically. Two identical runs
    differed by 7.9% on the tail-mean loss - the same size as the framework
    differences this probe exists to measure - which makes such a trace useless
    for attributing anything.
    """
    # cuBLAS reads this when it creates its handle, so it has to be set before
    # the first matmul or deterministic matmuls are silently not in effect.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Needed once anything routes through SDPA (the fp32 attention path does):
    # its flash and mem-efficient backends have nondeterministic backwards.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # warn_only: a few detection ops ship no deterministic kernel. Warn and keep
    # going rather than abort - then confirm empirically by running twice and
    # diffing the traces, which is the only check that actually proves it.
    torch.use_deterministic_algorithms(True, warn_only=True)


def apply_determinism(cfg: Config, args: argparse.Namespace) -> None:
    """Strip the configured sources of randomness from the config."""
    dataset_cfg = cfg.train_dataloader.dataset
    if args.ann_file is not None:
        dataset_cfg["ann_file"] = args.ann_file
    if not args.keep_augmentation:
        pipeline = []
        for entry in dataset_cfg.pipeline:
            if entry.get("type", "").split(".")[-1] in _AUGMENTATION_TYPES:
                continue
            if entry.get("type", "").endswith("ResizeCropFlipRotImage"):
                entry["training"] = False
                if "data_aug_conf" in entry:
                    entry["data_aug_conf"]["rand_flip"] = False
            pipeline.append(entry)
        dataset_cfg.pipeline = pipeline
    dataset_cfg["shuffle_cameras"] = False
    if not args.keep_grid_mask:
        cfg.model["use_grid_mask"] = False
    if not args.keep_dn:
        cfg.model.pts_bbox_head["with_dn"] = False


def build_fixed_batch(cfg: Config, args: argparse.Namespace) -> tuple[Any, list[str]]:
    """Collate a fixed batch through the framework's own dataloader machinery.

    Returns the batch alongside the sample tokens it was built from. This repo
    concatenates scenes sorted by ``scene_token`` while autoware-ml keeps the
    pkl order, so the same ``--start-index`` selects different frames on the
    two sides and only the tokens say so.
    """
    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    loader_cfg["batch_size"] = args.batch_size
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader_cfg["sampler"] = dict(type="DefaultSampler", shuffle=False)
    loader_cfg.pop("batch_sampler", None)
    dataloader = Runner.build_dataloader(loader_cfg, seed=args.seed)
    dataset = dataloader.dataset
    indices = list(range(args.start_index, args.start_index + args.batch_size))
    if indices[-1] >= len(dataset):
        raise ValueError(f"Indices {indices} exceed the {len(dataset)}-sample training split.")
    tokens = [str(dataset.get_data_info(index).get("token")) for index in indices]
    return dataloader.collate_fn([dataset[index] for index in indices]), tokens


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    """Collect every tensor in an arbitrarily nested batch value.

    Frameworks nest differently (AWML keeps a queue dimension, autoware-ml does
    not), so the fingerprint flattens before summarizing.
    """
    if isinstance(value, torch.Tensor):
        return [value]
    if hasattr(value, "tensor") and isinstance(getattr(value, "tensor"), torch.Tensor):
        return [value.tensor]  # mmdet3d box structure
    if isinstance(value, (list, tuple)):
        collected: list[torch.Tensor] = []
        for entry in value:
            collected.extend(flatten_tensors(entry))
        return collected
    if isinstance(value, np.ndarray):
        return [torch.from_numpy(value)]
    return []


# Lidar-frame points projected through every camera. Comparing the resulting
# pixel coordinates is independent of matrix layout and camera ordering, which
# a raw matrix mean is not.
_PROBE_POINTS = ((10.0, 0.0, 0.0), (20.0, 5.0, -1.0), (30.0, -5.0, 0.5))


def projection_probe(value: Any) -> list[list[float]] | None:
    """Project canonical lidar points through every camera matrix."""
    tensors = flatten_tensors(value)
    if not tensors:
        return None
    stacked = torch.cat([tensor.reshape(-1, 4, 4).float() for tensor in tensors], dim=0)
    pixels = []
    for matrix in stacked:
        for point in _PROBE_POINTS:
            projected = matrix[:3] @ torch.tensor([*point, 1.0])
            depth = float(projected[2])
            if depth <= 0.1:
                continue
            pixels.append(
                [round(float(projected[0]) / depth, 2), round(float(projected[1]) / depth, 2)]
            )
    return sorted(pixels)


def fingerprint_batch(batch: Any, tokens: list[str] | None = None) -> dict[str, Any]:
    """Summarize the batch with the same statistics as the autoware-ml probe."""
    report: dict[str, Any] = {}
    if tokens:
        # The one field that identifies the frame outright, so a frame mismatch
        # is a one-line read instead of an inference from gt_counts/timestamps.
        report["tokens"] = list(tokens)
    images = flatten_tensors(batch.get("img"))
    if images:
        image = torch.cat([tensor.reshape(-1, *tensor.shape[-3:]).float() for tensor in images])
        report["img"] = {
            "views": list(image.shape),
            "mean": round(float(image.mean()), 5),
            "std": round(float(image.std()), 5),
            "min": round(float(image.min()), 3),
            "max": round(float(image.max()), 3),
        }
    box_tensors = [
        tensor for tensor in flatten_tensors(batch.get("gt_bboxes_3d")) if tensor.dim() == 2
    ]
    if box_tensors:
        report["gt_counts"] = [int(tensor.shape[0]) for tensor in box_tensors]
        stacked = torch.cat([tensor.float() for tensor in box_tensors if tensor.numel()], dim=0)
        if stacked.numel():
            report["gt_box_mean"] = [round(float(v), 4) for v in stacked.mean(dim=0)[:7]]
    projection = projection_probe(batch.get("lidar2img"))
    if projection is not None:
        report["projected_pixels"] = projection
    stamps = flatten_tensors(batch.get("timestamp"))
    if stamps:
        values = torch.cat([tensor.reshape(-1).double() for tensor in stamps])
        # AWML stores sequence-relative stamps, autoware-ml absolute epoch
        # seconds, so only the within-batch deltas are comparable.
        report["timestamp_deltas"] = [round(float(v - values[0]), 3) for v in values[:8]]
    return report


def force_stream_reset(batch: Any) -> None:
    """Zero ``prev_exists`` so the head starts each step from empty memory."""
    value = batch.get("prev_exists")
    if isinstance(value, torch.Tensor):
        batch["prev_exists"] = torch.zeros_like(value)
    elif isinstance(value, (list, tuple)):
        batch["prev_exists"] = [torch.zeros_like(entry) for entry in value]


def build_wrapper(model: torch.nn.Module, cfg: Config, args: argparse.Namespace) -> Any:
    """Build the optimizer wrapper with a constant LR and the chosen precision."""
    wrapper_cfg = copy.deepcopy(cfg.optim_wrapper)
    if args.lr is not None:
        wrapper_cfg["optimizer"]["lr"] = args.lr
    elif cfg.get("auto_scale_lr", {}).get("enable", False):
        logger.warning(
            "auto_scale_lr is enabled in the config but this probe does not scale the LR. "
            "Pass --lr explicitly (the reference recipe's effective value is 1e-4) so both "
            "probes use the same value."
        )
    if args.precision == "fp32":
        wrapper_cfg["type"] = "OptimWrapper"
        wrapper_cfg.pop("loss_scale", None)
        wrapper_cfg.pop("dtype", None)
    else:
        wrapper_cfg["dtype"] = {"bf16": "bfloat16", "fp16": "float16"}[args.precision]
        if args.precision == "bf16":
            wrapper_cfg["loss_scale"] = dict(enabled=False)
    wrapper = build_optim_wrapper(model, wrapper_cfg)
    logger.info(
        "Optimizer groups: %s",
        [
            {"n": len(group["params"]), "lr": group["lr"]}
            for group in wrapper.optimizer.param_groups
        ],
    )
    return wrapper


def run_probe() -> None:
    """Overfit one fixed batch and write the per-step loss trace."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_random_seed(args.seed, deterministic=not args.nondeterministic)
    if args.nondeterministic:
        torch.backends.cudnn.benchmark = False
    else:
        enable_determinism()

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports") is not None:
        import_modules_from_strings(**cfg["custom_imports"])
    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    apply_determinism(cfg, args)

    batch, tokens = build_fixed_batch(cfg, args)
    fingerprint = fingerprint_batch(batch, tokens)
    logger.info("Batch fingerprint: %s", json.dumps(fingerprint, sort_keys=True))

    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, args.weights, map_location="cpu", strict=False, logger=logger)
    model.cuda().train()

    wrapper = build_wrapper(model, cfg, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        header = {
            "framework": "AWML",
            "config": args.config,
            "weights": args.weights,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "start_index": args.start_index,
            "precision": args.precision,
            "augmentation": args.keep_augmentation,
            "grid_mask": args.keep_grid_mask,
            "dn": args.keep_dn,
            "memory_carry": args.keep_memory,
            "deterministic": not args.nondeterministic,
            "fingerprint": fingerprint,
        }
        stream.write(json.dumps({"meta": header}) + "\n")

        for step in range(args.steps):
            if not args.keep_memory:
                model.pts_bbox_head.reset_memory()
            step_batch = copy.copy(batch)
            if not args.keep_memory:
                force_stream_reset(step_batch)
            log_vars = model.train_step(step_batch, wrapper)
            record = {
                "step": step,
                **{
                    key: round(float(value), 6)
                    for key, value in log_vars.items()
                    if isinstance(value, (int, float, torch.Tensor))
                },
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if step % 10 == 0 or step == args.steps - 1:
                logger.info("step %4d  loss=%.5f", step, record.get("loss", float("nan")))

    logger.info("Wrote %d steps to %s", args.steps, output_path)


if __name__ == "__main__":
    run_probe()

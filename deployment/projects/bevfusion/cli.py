"""BEVFusion CLI extensions."""

from __future__ import annotations

import argparse


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register BEVFusion-specific CLI flags onto a project subparser."""
    parser.add_argument(
        "--bevfusion-deploy-cfg",
        type=str,
        default=None,
        help="Path to BEVFusion mmdeploy-style deploy config (e.g. bevfusion_main_body_lidar_only_tensorrt_dynamic.py)",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="main_body",
        choices=["main_body", "image_backbone", "camera_bev_only"],
        help="Module to export (default: main_body)",
    )

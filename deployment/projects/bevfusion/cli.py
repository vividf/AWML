"""BEVFusion CLI extensions."""

from __future__ import annotations

import argparse


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register BEVFusion-specific CLI flags onto a project subparser."""
    parser.add_argument(
        "--module",
        type=str,
        default="main_body",
        choices=["main_body", "image_backbone", "camera_bev_only"],
        help="Module to export (default: main_body)",
    )

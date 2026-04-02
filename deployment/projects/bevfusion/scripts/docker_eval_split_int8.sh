#!/usr/bin/env bash
# Run split INT8 BEVFusion deploy eval inside awml-bevfusion:full.
# The image may not include pytorch-quantization; install it first (same as projects/BEVFusion/Dockerfile).
set -euo pipefail

IMAGE="${BEVFUSION_DOCKER_IMAGE:-awml-bevfusion:full}"
AWML_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker run --rm --gpus all --shm-size=32g \
  -v "${AWML_ROOT}:/workspace" \
  -v "${AWML_ROOT}/data:/workspace/data" \
  "${IMAGE}" \
  bash -lc '
set -e
python3 -m pip install --no-cache-dir \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple \
  pytorch-quantization==2.1.3
cd /workspace
export PYTHONPATH=/workspace:${PYTHONPATH:-}
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py \
  --module main_body \
  --log-level INFO
'

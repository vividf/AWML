"""BEVFusion-specific deployment runner."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.fx
from mmengine.config import Config

from deployment.configs import BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.core.device import DeviceSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.projects.bevfusion.eval.evaluator import BEVFusionEvaluator
from deployment.projects.bevfusion.export.onnx_export_pipeline import BEVFusionONNXExportPipeline
from deployment.projects.bevfusion.export.tensorrt_export_pipeline import BEVFusionTensorRTExportPipeline
from deployment.projects.bevfusion.io.model_loader import build_bevfusion_model
from deployment.runtime.runner import BaseDeploymentRunner

logger = logging.getLogger(__name__)


class BEVFusionDeploymentRunner(BaseDeploymentRunner):
    """BEVFusion deployment runner."""

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BEVFusionEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        module: str = "main_body",
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        self._module = module
        self._data_loader = data_loader

        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            onnx_wrapper_cls=IdentityWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

        if self._onnx_pipeline is None:
            self._onnx_pipeline = BEVFusionONNXExportPipeline(
                module=module,
                logger=self.logger,
            )

        if self._tensorrt_pipeline is None:
            self._tensorrt_pipeline = BEVFusionTensorRTExportPipeline(
                exporter_factory=ExporterFactory,
                components_cfg=config.components_cfg,
                logger=self.logger,
            )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        cuda_device = self.config.devices.cuda
        if cuda_device is None:
            raise RuntimeError(
                "BEVFusion requires a CUDA device for sparse convolution. " "Set devices.cuda in deploy config."
            )

        quantization = self.config.deploy_cfg.get("quantization", None)
        if quantization is not None:
            # Copy so we can safely mutate without touching the deploy_cfg singleton.
            quantization = dict(quantization)
            # Hoist top-level ``spconv_int8_fp16_layers`` (list of substring patterns matched on
            # sparse-conv module names) into the quantization dict. This is how model_loader
            # learns WHICH sparse convs to keep in FP16 — those modules get NO
            # _input_quantizer/_weight_quantizer, so the reloaded module tree matches the PTQ
            # checkpoint exactly (PTQ ran with the same exclusion).
            fp16_layers = self.config.deploy_cfg.get("spconv_int8_fp16_layers", None)
            if fp16_layers is not None:
                try:
                    quantization["spconv_int8_fp16_layers"] = list(fp16_layers)
                except TypeError:
                    quantization["spconv_int8_fp16_layers"] = []
        if quantization and quantization.get("enabled", False):
            logger.info("=" * 60)
            logger.info("BEVFusion INT8 Quantization Enabled")
            logger.info("  Dense (backbone/neck/head): pytorch_quantization")
            if quantization.get("spconv_int8", False):
                logger.info("  Sparse encoder: spconv INT8 (cumm kernels)")
            fp16_layers_ = quantization.get("spconv_int8_fp16_layers") or []
            if fp16_layers_:
                logger.info(
                    "  Sparse FP16 keep-list (spconv_int8_fp16_layers): %s",
                    fp16_layers_,
                )
            logger.info("=" * 60)

        model = build_bevfusion_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=cuda_device,
            quantization=quantization,
        )

        if quantization and quantization.get("enabled") and quantization.get("spconv_int8"):
            model = self._apply_spconv_int8(model, quantization, cuda_device)

        self.evaluator.set_pytorch_model(model)
        return model

    def _apply_spconv_int8(
        self,
        model: torch.nn.Module,
        quantization: dict,
        device: DeviceSpec,
    ) -> torch.nn.Module:
        """Apply spconv INT8 quantization to the sparse encoder.

        Uses manual calibration approach (no FX tracing) to work with
        BEVFusionSparseEncoder's complex forward method.
        """
        sparse_encoder = getattr(model, "pts_middle_encoder", None)
        if sparse_encoder is None:
            logger.warning("No pts_middle_encoder found; skipping spconv INT8")
            return model

        try:
            from deployment.projects.bevfusion.quantization.spconv_int8 import (
                apply_spconv_int8_quantization,
            )
        except ImportError as e:
            logger.warning(f"spconv quantization not available: {e}. Skipping spconv INT8.")
            return model

        torch_device = device.to_torch_device()

        # PTQ load already set up the encoder (FX GraphModule or NVIDIA TensorQuantizer).
        if isinstance(sparse_encoder, torch.fx.GraphModule):
            logger.info(
                "pts_middle_encoder is already an FX GraphModule (PTQ / model_loader path); "
                "skipping runner prepare_fx + calibrate + convert."
            )
            return model

        has_nvidia_quantizers = any(hasattr(m, "_input_quantizer") for m in sparse_encoder.modules())
        if has_nvidia_quantizers:
            logger.info(
                "pts_middle_encoder already has NVIDIA TensorQuantizer (PTQ / model_loader path); "
                "skipping runner FX calibrate + convert."
            )
            return model

        # Optional deploy override; PTQ checkpoints already carry an FX GraphModule (skipped above).
        num_calib_samples = int(quantization.get("num_calibration_samples", 40))
        calibration_data = self._collect_calibration_data(model, num_calib_samples, torch_device)

        if not calibration_data:
            logger.warning("No calibration data collected; skipping spconv INT8")
            return model

        try:
            from deployment.projects.bevfusion.quantization.spconv_int8 import (
                calibrate_spconv_model,
                convert_spconv_int8,
            )

            in_channels = getattr(sparse_encoder, "in_channels", 5)
            prepared = apply_spconv_int8_quantization(sparse_encoder, torch_device, in_channels=in_channels)
            calibrate_spconv_model(prepared, calibration_data)
            quantized_encoder = convert_spconv_int8(prepared, attr_source=sparse_encoder)
            model.pts_middle_encoder = quantized_encoder
            try:
                from deployment.projects.bevfusion.quantization.spconv_int8 import (
                    install_spconv_quantize_per_tensor_float_input_guard,
                )

                install_spconv_quantize_per_tensor_float_input_guard()
            except Exception:
                pass
            try:
                from projects.BEVFusion.bevfusion.bevfusion import register_pts_middle_encoder_float_input_hook

                register_pts_middle_encoder_float_input_hook(model.pts_middle_encoder)
            except Exception:
                pass
            logger.info("Spconv INT8 quantization applied to pts_middle_encoder")

        except Exception as e:
            logger.error(f"Spconv INT8 quantization failed: {e}")
            logger.info("Falling back to FP32 sparse encoder")
            import traceback

            traceback.print_exc()

        return model

    def _collect_calibration_data(
        self,
        model: torch.nn.Module,
        num_samples: int,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Collect calibration data by voxelizing sample point clouds.

        Returns list of (voxel_features, coors, batch_size) tuples
        suitable for the sparse encoder's forward method.
        """
        calibration_data = []
        actual_samples = min(num_samples, self._data_loader.num_samples)

        logger.info(f"Collecting {actual_samples} calibration samples for spconv INT8...")

        for idx in range(actual_samples):
            try:
                sample = self._data_loader.load_sample(idx)
                points = sample.get("points", None)
                if points is None:
                    continue

                if not isinstance(points, torch.Tensor):
                    points = torch.from_numpy(points)
                points = points.to(device).float()

                with torch.no_grad():
                    ret = model.pts_voxel_layer(points)
                    if len(ret) == 3:
                        feats, coords, sizes = ret
                    else:
                        feats, coords = ret
                        sizes = None

                    batch_coors = torch.zeros(coords.shape[0], 1, device=device, dtype=coords.dtype)
                    coords = torch.cat([batch_coors, coords], dim=1).contiguous()

                    if sizes is not None and getattr(model, "voxelize_reduce", True):
                        feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                        feats = feats.contiguous()

                    calibration_data.append((feats, coords.int(), 1))
                    logger.debug(f"  Calibration sample {idx}: {feats.shape[0]} voxels")

            except Exception as e:
                logger.warning(f"  Failed to load calibration sample {idx}: {e}")
                continue

        logger.info(f"Collected {len(calibration_data)} calibration samples")
        return calibration_data

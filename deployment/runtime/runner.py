"""
Unified deployment runner for common deployment workflows.

Project-agnostic runtime runner that orchestrates:
- Export (PyTorch -> ONNX -> TensorRT)
- Verification (scenario-based comparisons)
- Evaluation (metrics/latency across backends)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from mmengine.config import Config

from deployment.configs.base import BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.core.evaluation.backend_executor import BackendExecutor
from deployment.core.evaluation.backend_verifier import BackendVerifier
from deployment.core.evaluation.base_evaluator import BaseEvaluator
from deployment.core.evaluation.output_comparator import OutputComparator
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.runtime.artifact_manager import ArtifactManager
from deployment.runtime.evaluation_orchestrator import EvaluationOrchestrator
from deployment.runtime.export_orchestrator import ExportOrchestrator
from deployment.runtime.verification_orchestrator import VerificationOrchestrator


@dataclass
class DeploymentResult:
    """Standardized structure returned by `BaseDeploymentRunner.run()`."""

    pytorch_model: Optional[Any] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)


class BaseDeploymentRunner:
    """Base deployment runner for common deployment pipelines."""

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        executor: BackendExecutor,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_wrapper_cls: Optional[Type[BaseModelWrapper]] = None,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        self.data_loader = data_loader
        self.evaluator = evaluator
        self._executor = executor
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger

        self.artifact_manager = ArtifactManager(config, logger)

        self.export_orchestrator = ExportOrchestrator(
            config=config,
            data_loader=data_loader,
            artifact_manager=self.artifact_manager,
            logger=logger,
            model_loader=self.load_pytorch_model,
            onnx_wrapper_cls=onnx_wrapper_cls,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )
        comparator = OutputComparator(output_names=executor.get_output_names())
        verifier = BackendVerifier(executor, comparator)
        self.verification_orchestrator = VerificationOrchestrator(
            config, verifier, data_loader, self.artifact_manager, logger
        )
        self.evaluation_orchestrator = EvaluationOrchestrator(
            config, evaluator, data_loader, self.artifact_manager, logger
        )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.load_pytorch_model() must be implemented by subclasses.")

    def run(self, context: Optional[ExportContext] = None) -> DeploymentResult:
        if context is None:
            context = ExportContext()

        results = DeploymentResult()

        export_result = self.export_orchestrator.run(context)
        results.pytorch_model = export_result.pytorch_model
        results.onnx_path = export_result.onnx_path
        results.tensorrt_path = export_result.tensorrt_path

        # Hand the loaded reference model to the executor shared by verification and evaluation.
        self._executor.set_pytorch_model(export_result.pytorch_model)

        results.verification_results = self.verification_orchestrator.run()
        results.evaluation_results = self.evaluation_orchestrator.run()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Deployment Complete!")
        self.logger.info("=" * 80)

        return results

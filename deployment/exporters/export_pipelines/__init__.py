"""Export pipelines.

``OnnxExportPipeline`` (``onnx_pipeline``) and ``TensorRTExportPipeline``
(``tensorrt_pipeline``) are the concrete pipelines. Model-specific variation is
injected via two seams, each module pairing its interface with the built-in
default implementation: ``sample_adapter`` (``ExportSampleAdapter`` +
``DefaultSampleAdapter``) and ``component_builder`` (``ModelComponentBuilder`` +
``DefaultComponentBuilder``).
"""

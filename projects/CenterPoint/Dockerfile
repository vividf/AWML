ARG AWML_BASE_IMAGE="autoware-ml:latest"
FROM ${AWML_BASE_IMAGE}
ARG TRT_VERSION=10.8.0.43

# Install pip dependencies
RUN python3 -m pip --no-cache-dir install \
  onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxsim \
  pycuda \
  tensorrt-cu12==${TRT_VERSION}

RUN pip install --no-cache-dir --index-url https://pypi.nvidia.com --index-url https://pypi.org/simple pytorch-quantization==2.1.3

WORKDIR /workspace
RUN pip install --no-cache-dir -e .
nt8_resnet_deployment/onnx/pts_voxel_encoder.onnx
I0126 06:55:07.503386 139622071670592 onnx_export_pipeline.py:132]
[2/2] Exporting pts_backbone_neck_head...
I0126 06:55:07.503471 139622071670592 base_exporter.py:64] Applying model wrapper for export
I0126 06:55:07.504007 139622071670592 onnx_exporter.py:155] Exporting model to ONNX format...
I0126 06:55:07.504039 139622071670592 onnx_exporter.py:157]   Input shape: torch.Size([1, 32, 1020, 1020])
I0126 06:55:07.504056 139622071670592 onnx_exporter.py:158]   Output path: work_dirs/centerpoint_int8_resnet_deployment/onnx/pts_backbone_neck_head.onnx
I0126 06:55:07.504070 139622071670592 onnx_exporter.py:159]   Opset version: 16
W0126 06:55:07.506877 139622071670592 tensor_quantizer.py:281] Use Pytorch's native experimental fake quantization.
/opt/conda/lib/python3.11/site-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if amax.numel() == 1:
/opt/conda/lib/python3.11/site-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:286: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  inputs, amax.item() / bound, 0,
/opt/conda/lib/python3.11/site-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:292: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  quant_dim = list(amax.shape).index(list(amax_sequeeze.shape)[0])

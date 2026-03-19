# Docker Initial Setup Plan for `.so` Dependencies

This README defines a practical setup plan to make sure your Docker image/container has the required shared libraries for BEVFusion TensorRT deployment, especially:

- `libspconv.so` (from `spconv_cpp` or equivalent pure C++ build)
- TensorRT custom plugin `.so` that registers:
  - `ImplicitGemm`
  - `GetIndicePairsImplicitGemm`

> `pip install spconv-cu120` is useful for Python training/inference, but it does **not** guarantee TensorRT parser can find the custom plugin creators above.

---

## 1) Target Outcome

After setup, all checks below should pass in container:

1. `libspconv.so` can be found by dynamic linker.
2. Custom TensorRT plugin `.so` can be loaded (`ctypes.CDLL(..., RTLD_GLOBAL)`).
3. TensorRT registry contains plugin creators:
   - `ImplicitGemm`
   - `GetIndicePairsImplicitGemm`
4. AWML `tensorrt_config.plugin_libraries` points to the plugin `.so`.

---

## 2) Version Alignment (Before Build)

Make sure the stack is compatible:

- CUDA version in base image
- TensorRT runtime version in image
- `spconv_cpp` variant/version
- custom TensorRT plugin build target (must match TensorRT/CUDA ABI)

If versions are mixed, parser usually fails with:
`Plugin not found, are the plugin name, version, and namespace correct?`

---

## 3) Dockerfile Plan (Image Build Stage)

Use this as the initial setup pattern in your Dockerfile.

```dockerfile
# 1) Build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2) Build and install spconv_cpp -> cumm + spconv deb packages
WORKDIR /opt
RUN git clone https://github.com/autowarefoundation/spconv_cpp.git

RUN cd /opt/spconv_cpp && \
    mkdir -p cumm/build-amd64 && \
    cd cumm/build-amd64 && \
    cmake .. && make -j"$(nproc)" && cpack -G DEB && \
    apt-get update && apt-get install -y /opt/spconv_cpp/cumm/_packages/cumm_0.5.3_amd64.deb

RUN cd /opt/spconv_cpp && \
    mkdir -p spconv/build-amd64 && \
    cd spconv/build-amd64 && \
    cmake .. && make -j"$(nproc)" && cpack -G DEB && \
    apt-get update && apt-get install -y /opt/spconv_cpp/spconv/_packages/spconv_2.3.8_amd64.deb

# 3) Refresh linker cache
RUN ldconfig
```

---

## 4) Add TensorRT Custom Plugin `.so`

You still need the custom TensorRT plugin library that registers sparse conv creators.

Example layout:

- `/opt/plugins/libautoware_tensorrt_plugins.so` (example name)

Then in Dockerfile:

```dockerfile
COPY path/on/host/libautoware_tensorrt_plugins.so /opt/plugins/libautoware_tensorrt_plugins.so
RUN chmod 755 /opt/plugins/libautoware_tensorrt_plugins.so && ldconfig
```

If this plugin `.so` is built from another repository, build it in a dedicated stage and copy artifact into runtime image.

---

## 5) Runtime Verification Commands

Enter container:

```bash
docker exec -it awml-centerpoint bash
```

Check linker visibility:

```bash
ldconfig -p | rg -i "spconv|nvinfer|plugin"
```

Check plugin creators in TensorRT:

```bash
python - <<'PY'
import ctypes
import tensorrt as trt

PLUGIN_SO = "/opt/plugins/libautoware_tensorrt_plugins.so"  # adjust path
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(logger, "")

reg = trt.get_plugin_registry()
for name in ("ImplicitGemm", "GetIndicePairsImplicitGemm"):
    hits = [c for c in reg.plugin_creator_list if c.name == name]
    print(name, "count =", len(hits), [(c.plugin_version, c.plugin_namespace) for c in hits])
PY
```

Expected: both counts are `>= 1`.

---

## 6) AWML Deployment Config Wiring

Set plugin path in:

- `deployment/projects/bevfusion/config/deploy_config.py`

```python
tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=1 << 32,
    plugin_libraries=[
        "/opt/plugins/libautoware_tensorrt_plugins.so",
    ],
)
```

You can also use env vars:

- `DEPLOY_TENSORRT_PLUGIN_LIBS`
- `TENSORRT_PLUGIN_LIBS`

---

## 7) Step-by-Step Execution Plan (Recommended)

1. Build image with `spconv_cpp` (`cumm` + `spconv`) and install deb packages.
2. Add/copy TensorRT custom plugin `.so` into image.
3. Run container and execute verification script (Section 5).
4. Configure `plugin_libraries` in AWML deploy config.
5. Run TensorRT export again.
6. If still failing, verify exact plugin creator `name/version/namespace` against ONNX parser error logs.

---

## 8) Common Failure Patterns

- `spconv-cu120` installed, but no TensorRT creator registered  
  -> Python wheel is present, but TensorRT custom plugin `.so` is missing or not loaded.

- `ctypes.CDLL` fails with missing dependency  
  -> fix `LD_LIBRARY_PATH` / install missing `.so` / run `ldconfig`.

- Creator exists but parser still fails  
  -> plugin creator `version/namespace` mismatch with ONNX node expectation.

---

## 9) Minimal Acceptance Checklist

- [ ] `libspconv.so` visible in `ldconfig -p`
- [ ] plugin `.so` file exists at configured path
- [ ] `ctypes.CDLL(plugin_so, RTLD_GLOBAL)` succeeds
- [ ] TensorRT registry shows `ImplicitGemm` and `GetIndicePairsImplicitGemm`
- [ ] AWML `plugin_libraries` is configured
- [ ] ONNX -> TensorRT export no longer fails at parser stage

# BEVFusion TensorRT Export: End-to-End Debug Guide

這份文件記錄這次從失敗到成功的完整流程，目標是讓你在新環境也能重現。

## TL;DR (最短成功路徑)

1. 在 BEVFusion container 內編譯並安裝 plugin：
   - `bash projects/BEVFusion/plugins/build_plugin_inside_container.sh`
2. 驗證 creators：
   - `python3 projects/BEVFusion/deploy/check_trt_spconv_plugins.py --plugin-so /opt/plugins/libautoware_tensorrt_plugins.so`
3. 設定 plugin 載入：
   - `export DEPLOY_TENSORRT_PLUGIN_LIBS=/opt/plugins/libautoware_tensorrt_plugins.so`
4. 修正 TensorRT profile（`voxels` 第 3 維必須是 `5`）
5. 重跑 export 指令

---

## 1) 問題背景與關鍵錯誤

最初失敗訊息：

- `Plugin not found ... operator: ImplicitGemm`
- ONNX parser 無法通過，`TensorRT export failed: unable to parse ONNX file`

結論：

- BEVFusion ONNX 需要 sparse conv plugin creators：
  - `ImplicitGemm`
  - `GetIndicePairsImplicitGemm`
- 這兩個 creator 不在 TensorRT 內建，必須提供自訂 `.so`。

---

## 2) 這次最終成功所依賴的檔案

- `projects/BEVFusion/plugins/build_plugin_inside_container.sh`
  - 在容器內建置 plugin
- `projects/BEVFusion/plugins/CMakeLists.standalone`
  - 脫離 ROS/ament 的 standalone CMake
- `projects/BEVFusion/deploy/check_trt_spconv_plugins.py`
  - 驗證 plugin creators
- `deployment/core/tensorrt_plugins.py`
  - export/runtime 載入 plugin 的統一入口
- `projects/BEVFusion/configs/deploy/bevfusion_main_body_lidar_only_tensorrt_dynamic.py`
  - TensorRT model_inputs shape
- `deployment/projects/bevfusion/config/deploy_config.py`
  - TensorRT profile、plugin 設定

---

## 3) 從頭操作步驟

### Step A: 啟動容器

```bash
docker run -it --rm --gpus all --shm-size=32g \
  --name awml-bevfusion \
  -p 6007:6007 \
  -v $PWD/:/workspace \
  -v $PWD/data:/workspace/data \
  awml-bevfusion:full
```

### Step B: 容器內編譯 plugin

```bash
cd /workspace
bash projects/BEVFusion/plugins/build_plugin_inside_container.sh
```

成功後應看到：

- `Installed: /opt/plugins/libautoware_tensorrt_plugins.so`

### Step C: 驗證 creators

```bash
python3 projects/BEVFusion/deploy/check_trt_spconv_plugins.py \
  --plugin-so /opt/plugins/libautoware_tensorrt_plugins.so
```

成功案例（本次）：

- `ImplicitGemm: 1`
- `GetIndicePairsImplicitGemm: 1`
- `All required sparse creators are available.`

### Step D: 設定 plugin 載入

二選一：

1) 用環境變數（建議 quick test）

```bash
export DEPLOY_TENSORRT_PLUGIN_LIBS=/opt/plugins/libautoware_tensorrt_plugins.so
```

2) 或在 config 指定：

- `deployment/projects/bevfusion/config/deploy_config.py`
  - `tensorrt_config.plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"]`

### Step E: 檢查/修正 TensorRT profile

這次踩到的錯誤：

- `Dimension mismatch for tensor voxels ... profile axis 2 = 4 but tensor has 5`

修正為：

- `voxels` profile 要用 `[*, 10, 5]`，不是 `[*, 10, 4]`

已調整：

- `projects/BEVFusion/configs/deploy/bevfusion_main_body_lidar_only_tensorrt_dynamic.py`
- `deployment/projects/bevfusion/config/deploy_config.py`

### Step F: 重新 export

```bash
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
  --module main_body
```

---

## 4) 這次實際 debug 過的錯誤與修法

### 錯誤 1: `Plugin not found (ImplicitGemm)`

原因：

- 容器裡沒有可註冊該 creator 的 plugin `.so`。

修法：

- 在容器內建出 `libautoware_tensorrt_plugins.so`
- 載入 plugin 後再 parse ONNX

---

### 錯誤 2: `NVINFER / NVONNXPARSER NOTFOUND`

原因：

- pip TensorRT 常只有 versioned `.so`，`find_library(nvinfer)` 不一定找得到。

修法：

- 在 script 裡自動掃描 `tensorrt` / `tensorrt_libs` 實際 `.so`，顯式傳給 CMake。

---

### 錯誤 3: `--extended-lambda` 缺失

原因：

- CUDA flags 沒真正掛到 target compile options。

修法：

- 在 `CMakeLists.standalone` 以 `target_compile_options(... COMPILE_LANGUAGE:CUDA ...)` 明確加上 `--extended-lambda`。

---

### 錯誤 4: `NvInferRuntime.h: No such file or directory`

原因：

- 容器有 TensorRT runtime libs，但缺 C++ headers。

修法：

- script 自動嘗試安裝 `libnvinfer-dev` / `libnvonnxparsers-dev`，並尋找 header include path。

---

### 錯誤 5: `.so` 載入成功但 creators 還是 0

原因：

- TensorRT 10 plugin 註冊需要優先透過 registry `load_library`，僅 `ctypes.CDLL` 可能看不到 creators。

修法：

- `deployment/core/tensorrt_plugins.py` 與檢查腳本改為優先用 registry 載入（runtime / builder）。

---

### 錯誤 6: `profile axis mismatch (voxels dim=4 vs 5)`

原因：

- TensorRT profile 與 ONNX 實際輸入 shape 不一致。

修法：

- 把 `voxels` 的 profile 第 3 維改成 `5`。

---

## 5) 建議的日常檢查清單

每次換模型或資料集，建議先做：

1. `check_trt_spconv_plugins.py` 確認 creators 存在
2. 確認 `plugin_libraries` 或 `DEPLOY_TENSORRT_PLUGIN_LIBS` 已生效
3. 檢查 deploy profile 與 ONNX 實際 input shape 一致（尤其 feature 維度）
4. 再跑 engine build

---

## 6) 補充

- `spconv-cu120`（Python wheel）不等於 TensorRT plugin creators 可用。
- `spconv_cpp` 主要提供 `libspconv.so` / `cumm`，不是完整 TRT parser plugin 註冊。
- 若 plugin creators 已可見但 parser 仍失敗，下一步檢查 `name/version/namespace` 是否完全對齊。

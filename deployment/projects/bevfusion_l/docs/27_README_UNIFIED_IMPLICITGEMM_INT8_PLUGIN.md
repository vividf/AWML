# 27. 統一 ImplicitGemm Plugin（單一 plugin 同時支援 FP16 與 INT8）

## 背景與目標

原本 INT8 sparse convolution 走的是一份獨立的 plugin：
`AWML/deployment/projects/bevfusion_l/cpp/int8_plugin/`（`ImplicitGemmInt8`），
而 FP16 走的是 `autoware.universe` 內既有的 `ImplicitGemm`
（`perception/autoware_tensorrt_plugins/`）。兩份程式碼大量重複。

本次重構的目標：

- **不要 duplicate file**。把 INT8 真正需要的東西併進 `autoware.universe` 既有的
  `ImplicitGemm` plugin，讓 **同一顆 plugin（同一個 `.so`、同一個 op 名稱）** 同時支援 FP16 與 INT8。
- ONNX 端 **不改 op 名稱**（維持 `op_type="ImplicitGemm"`），改用 **`precision` 屬性**
  （`0=FP`、`1=INT8`）讓 plugin 在 runtime 自行判斷走哪條路徑。
- 在 container 內 build 出單一 `libautoware_tensorrt_plugins.so` 即可使用 INT8。

> 註：所有 timing / CUDA-event profiling 相關程式碼已全部移除（plugin、creator、ONNX 屬性、
> deploy config keys、audit 都不再有 timing 欄位）。本文件描述的是移除 timing 之後的最終狀態。

---

## 改了什麼（autoware.universe，單一 plugin 支援 INT8）

`perception/autoware_tensorrt_plugins/`：

### `include/autoware/tensorrt_plugins/implicit_gemm_plugin.hpp`

- `ImplicitGemmParameters` 新增：
  - `precision`（`0=FP`、`1=INT8`；常數 `kIMPLICIT_GEMM_PRECISION_FP`/`kIMPLICIT_GEMM_PRECISION_INT8`）
  - `input_scale`（INT8 量化 features 用，= `input_amax / 127`）
  - struct 為 POD，整顆序列化（plugin 的 serialize 機制不變）。
- class 新增 INT8 用成員：
  - `tuner_int8_ptr_`（INT8 ConvTuner）
  - 權重快取指標 `cached_weight_int8_ptr_` / `cached_w_scales_ptr_` / `cached_gemm_bias_ptr_`
    + 對應的 `cached_c_out_/k1/k2/k3/c_in_` 與 `cache_allocated_`、`cache_filled_` 旗標
  - `cache_mutex_`（保護 cache 配置/填充）
  - 輸入索引常數：`INOUT_CHANNEL_SCALE_INDEX=5`、`INOUT_BIAS_SCALED_INDEX=6`、`NUM_INPUTS_INT8=7`
  - `is_int8()` 與 INT8 helper 宣告（`enqueueInt8` / `allocateConstantCache` / `releaseConstantCache`）。
- 析構改為自訂（釋放權重快取）。

### `src/implicit_gemm_plugin.cpp`

- `enqueue` 開頭：`if (is_int8()) return enqueueInt8(...);`，**FP16 路徑一字未動**。
- `configurePlugin` / `supportsFormatCombination` / `getOutputDataTypes` / `getOutputShapes` /
  `onShapeChange` / `getWorkspaceSize` 都加上 **7-input 的 INT8 分支**：
  - 7 inputs：FP16 features / FP16 filters / INT32 pair_fwd / INT32 pair_mask /
    INT32 mask_argsort / FP32 channel_scale (idx 5) / FP32 bias_scaled (idx 6)；輸出 FP16。
- 新增 `enqueueInt8`：
  1. 把 FP16 features 量化成 INT8（寫進 workspace scratch）。
  2. 第一次 enqueue 時把權重/scale/bias 量化填入持久 cache（filter device data 只有在 enqueue 才拿得到）。
     epilogue 用 `s8s8f16`，因此 `output_scale` 折進 scale/bias，GEMM 的 `output_scale` 傳 1。
  3. 呼叫 `ConvGemmOps::implicit_gemm`（INT8 in、FP16 out）。
- `allocateConstantCache` / `releaseConstantCache`：
  - cache 的 `cudaMalloc` 放在 `configurePlugin`（build）與 `onShapeChange`（runtime），
    **避開 CUDA graph capture 期間 malloc**；kernel fill 留在第一次 enqueue。
  - `enqueueInt8` 內另保留一個 fallback 配置路徑（兩者都沒先跑到時的保險），以 `cache_mutex_` 保護。

### `src/quantize_ops/quantize_features.cu` + `include/autoware/quantize_ops/quantize_features.hpp`

INT8 量化 kernels（已加入 `cuda_ops`）：

- `launch_quantize_features`：FP16 features → INT8。
- `launch_quantize_weights_per_channel`：FP16 filters → INT8（per-output-channel）。
- `launch_compute_w_scales`：計算 per-channel weight scale。
- `launch_fuse_output_scale_into_gemm_scale_bias`：把 `output_scale` 折進 GEMM 的 scale/bias
  （配合 `s8s8f16` epilogue 不乘 alpha 的行為）。

### `src/implicit_gemm_plugin_creator.cpp`

- build phase 解析 `precision` / `input_scale`（`num_fields >= 6`，未知欄位忽略）。
- `plugin_attributes_` 表同步擴充 `precision` / `input_scale`。
- **不需要新增第二個 creator / 第二個 plugin 名稱**。

### `CMakeLists.txt`

- `cuda_ops` 加入 `src/quantize_ops/quantize_features.cu`。

### `plugin_registration.cpp`

- **不用動**（沒有新 op 名稱）。

---

## 改了什麼（AWML）

- `export/sparse_int8_onnx_transform.py`：
  - 轉換後的節點 **保留 `op_type="ImplicitGemm"`** 並加上 **`precision=1`** 屬性
    （不再改名成 `ImplicitGemmInt8`），維持 7 inputs。
  - 新增 `_implicit_gemm_node_precision`；轉換 idempotent（已轉過的節點會跳過）；
    census 改用 `precision` 區分 FP16 / INT8。
- `export/sparse_int8_onnx_audit.py`、`debug/compare_sparse_onnx_pt_weights.py`：
  改以 `op_type="ImplicitGemm"` + `precision==1` 偵測 INT8 節點。
- `config/deploy_config_split_sparse_int8_dense_int8.py`：
  `plugin_libraries` 移除獨立的 `libimplicit_gemm_int8_plugin.so`，
  只留 `libautoware_tensorrt_plugins.so`。

---

## 驗證狀態

- AWML Python 全部語法檢查 OK（`ast.parse`）。
- 在 AWML container（`awml-bevfusion:full`）內用 standalone CMake 完整編譯通過：
  `libautoware_tensorrt_plugins.so` + `libcuda_ops.so`（含新的 `quantize_features.cu`）皆成功產出。
- IDE 顯示的 clangd 錯誤都是缺 include path 的假警報（既有的 FP16 plugin 在 container 外也一樣）。

---

## 在哪個 container build？兩條路

`autoware_tensorrt_plugins/CMakeLists.txt` 是 **ament_cmake / autoware_cmake** package，
正規做法要在 ROS 2 / Autoware 的 colcon workspace 內 build。但 plugin 的 **原始碼本身**
（`src/`、`include/`）**完全不依賴 ROS / ament**，只需要 TensorRT + CUDA + spconv / cumm，
而這些在 AWML 部署 image（`awml-bevfusion:full`）內都有。

確認方式（container 內）：

```bash
which colcon; echo "ROS_DISTRO=$ROS_DISTRO"   # 通常為空 → 沒有 ROS/colcon
find / -name spconvConfig.cmake 2>/dev/null    # 有 → spconv 在
find / -name cummConfig.cmake   2>/dev/null    # 有 → cumm 在
```

### 路線 A（推薦，無需 commit 到 github）：在 AWML container 內用 standalone build

AWML 已內建 `projects/BEVFusion/plugins/build_plugin_inside_container.sh`
+ `projects/BEVFusion/plugins/CMakeLists.standalone`，用 **純 CMake**（無 ament）編出
相同的 `libautoware_tensorrt_plugins.so`，並 install 到 `/opt/plugins/`。
（`CMakeLists.standalone` 已加入 `quantize_ops/quantize_features.cu`，否則 INT8 會 link 失敗。）

該腳本支援用 **bind-mount 的本地 source**（`AUTOWARE_TENSORRT_PLUGINS_SRC`），會 **跳過 git clone**
—— 這正是避免「一直 commit 到 github」的關鍵：你本地對 `autoware.universe` 的修改透過 bind mount
立刻可見，不必 push/pull。

1. 啟動 container 時，把 sibling 的 `autoware.universe` 一起掛進去：

```bash
docker run -it --rm --gpus all --shm-size=32g --name awml-bevfusion -p 6007:6007 \
  -v $PWD/:/workspace \
  -v $PWD/data:/workspace/data \
  -v ~/ml_workspace/autoware.universe:/workspace/autoware.universe \
  awml-bevfusion:full
```

2. container 內 build（指向掛進來的本地 source，跳過 clone）：

```bash
export AUTOWARE_TENSORRT_PLUGINS_SRC=/workspace/autoware.universe/perception/autoware_tensorrt_plugins
bash projects/BEVFusion/plugins/build_plugin_inside_container.sh
# 產物：/opt/plugins/libautoware_tensorrt_plugins.so（deploy config 指向此路徑）
```

> 改完 plugin 程式碼後，重跑同一行即可重編；因為是 bind mount，不需 commit / push。

### 路線 B（正規 CI 路徑）：在 Autoware colcon workspace build

```bash
cd <autoware_ws>   # autoware.universe 所在的 colcon workspace
colcon build --packages-select autoware_tensorrt_plugins \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
# 產物：
#   install/autoware_tensorrt_plugins/share/autoware_tensorrt_plugins/plugins/libautoware_tensorrt_plugins.so
# 若 deploy config 指向 /opt/plugins/...，把上面的 .so 覆蓋過去
```

---

## 接著做（兩條路共用）

### 重匯出 INT8 ONNX（必須，因為 op 屬性變了）

```bash
python -m deployment.projects.bevfusion_l.export.sparse_int8_onnx_transform \
  --onnx <fp16 sparse onnx> \
  --checkpoint vivid/bench_comparison/bevfusion_2_7/best_epoch_28_ptq_sparse_int8_dense_int8.pth \
  --deploy-cfg deployment/projects/bevfusion_l/config/deploy_config_split_sparse_int8_dense_int8.py \
  --output <work_dir>/onnx/bevfusion_sparse.onnx
```

### 重建 sparse engine + 評測

```bash
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion_l/config/deploy_config_split_sparse_int8_dense_int8.py \
  projects/BEVFusion/configs/.../bevfusion_..._120m.py
```

---

## 為什麼用 `precision` 屬性而不是新 op 名稱

- TensorRT 是用 ONNX `op_type` 對應到註冊的 plugin creator。若改名成 `ImplicitGemmInt8`，
  就得維護第二個 creator + 第二份 plugin class，回到 duplicate 的老路。
- 改用 `precision` 屬性後：ONNX graph 更乾淨、plugin registry 只需要一個 `ImplicitGemm`，
  FP16 與 INT8 共用同一份 lifecycle 程式碼，差異只集中在 `enqueueInt8` 與 cache 配置。
- 代價：INT8 ONNX 需要 **重新匯出**（因為節點屬性從舊的 `ImplicitGemmInt8` 改成
  `ImplicitGemm` + `precision=1`）。

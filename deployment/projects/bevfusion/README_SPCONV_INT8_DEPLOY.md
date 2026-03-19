# BEVFusion Deployment 與 Spconv INT8 使用方式

在現有 deployment 指令下：

```bash
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m.py \
  --module main_body
```

若要使用 **INT8 spconv**（稀疏卷積用 INT8 計算），可以採用以下兩種方式，依需求選擇。

---

## 方式一：PyTorch backend + PTQ 量化模型（推薦，真正 spconv INT8）

**概念**：在 framework 外先用 spconv 的 PTQ 流程得到「量化後的 PyTorch 模型」，deployment 時用 **PyTorch backend** 載入該 checkpoint，推論時稀疏卷積會走 cumm 的 INT8 kernel。

**步驟概要**：

1. **在專案外做 PTQ 並存成 checkpoint**
   - 載入原本的 BEVFusion main_body（或至少含 SCN 的 subgraph）。
   - 使用 `spconv.pytorch.quantization`：`prepare_fx` → 用代表性資料 **calibrate** → `convert_fx` → `transform_qdq` → `remove_conv_add_dq`（參考 `spconv/test/develop/mnist_int8_dev.py`）。
   - 將轉好的 quantized 模型存成 checkpoint（例如 `bevfusion_main_body_ptq.pth`）。

2. **在 deployment 中改為載入 PTQ checkpoint**
   - 在 config 或 CLI 中支援「quantized checkpoint 路徑」（例如 `checkpoint_path` 指向 `bevfusion_main_body_ptq.pth`）。
   - 使用 **PyTorch** backend 時，載入該 checkpoint 而非原本的 FP32/FP16 權重。

3. **執行時指定 PyTorch backend**
   - 執行 evaluation/run 時後端選 `pytorch`，這樣會走 `BEVFusionPyTorchPipeline`，推論即為 spconv INT8。

**優點**：真正使用 spconv INT8（cumm kernel），不需改 ONNX / TensorRT plugin。  
**缺點**：需實作「載入 quantized checkpoint」的邏輯；PyTorch 推論延遲一般高於 TensorRT。

**目前 framework 狀態**：尚未內建「PTQ 導出 + 載入 quantized checkpoint」一鍵流程；需要自行在外部完成 PTQ 並存檔，再在 deployment 中把 `checkpoint_path` 指到該檔案並使用 `--backend pytorch`（或你們實際的後端參數）。

---

## 方式二：TensorRT INT8（非 spconv 自帶 INT8）

**概念**：維持現有流程「ONNX 導出 → TensorRT engine」，在 `deploy_config.py` 的 `tensorrt_config` 裡開啟 TensorRT 的 FP16/INT8，讓 TensorRT 盡量用低精度。

**設定**：在 `deploy_config.py` 的 `tensorrt_config` 中加入或修改為：

```python
tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=1 << 32,
    policy_flags=dict(FP16=True, INT8=True),  # 開啟 TRT INT8（若需 calibration 需另設）
    plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"],
)
```

**注意**：BEVFusion 的稀疏卷積是透過 **TensorRT 自訂 plugin**（ImplicitGemm / GetIndicePairsImplicitGemm）執行。這些 plugin 是否支援 INT8 取決於 plugin 實作；多數情況下 sparse 部分仍是 FP16/FP32。因此這是「TensorRT 的 INT8」（對 dense 等層有效），不保證是「spconv 的 INT8」。

---

## 對照與建議

| 目標                     | 建議方式                         |
|--------------------------|----------------------------------|
| 要 **spconv 層** 真的跑 INT8 | 方式一：PyTorch backend + PTQ checkpoint |
| 要 **TensorRT 整體** 加速/省記憶體 | 方式二：TensorRT + policy_flags INT8/FP16 |
| 要與現有 `python -m deployment.cli.main bevfusion ... --module main_body` 一致 | 方式一需擴充「載入 PTQ checkpoint」；方式二只需改 config 建 engine |

若目標是「在現有 deployment framework 裡使用 **spconv INT8**」，應採用 **方式一**，並在專案中新增或擴充：

- 支援從 config/CLI 指定 quantized checkpoint 路徑；
- 當使用 PyTorch backend 時，載入該 PTQ 模型而非原始 checkpoint。

spconv PTQ 的實作細節可參考 Lidar_AI_Solution 中的 `README_SPCONV_INT8.md` 與 spconv 的 `test/develop/mnist_int8_dev.py`。

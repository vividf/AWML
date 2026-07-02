# BEVFusion：ONNX、TensorRT、spconv INT8 與 libspconv — 部署路線總整理

本文彙整 **AWML BEVFusion** 在 **PTQ / spconv INT8** 情境下，與 **ONNX、TensorRT、spconv 官方文件、Lidar AI Solution 類拆分** 的關係，並回答常見問題：

- 怎樣做比較可能「正確」得到 `.onnx` 並接 TensorRT？
- **spconv `gencode` 要不要重做？要不要改 spconv 原始碼？**
- **libspconv 怎麼吃 PTQ 的 int8 sparse 權重／scale？**
- **是否應像 Lidar AI Solution 一樣拆成多個 ONNX？**

更貼近程式修改的實作紀錄另見：  
[4_spconv_int8_implementation_history_zh.md](./4_spconv_int8_implementation_history_zh.md)。

**Split + PTQ INT8 進度、除錯時間線、明日接續點**：  
[6_bevfusion_split_ptq_int8_progress.md](./6_bevfusion_split_ptq_int8_progress.md)。

---

## 〇、與 Lidar AI Solution 對齊（原則與路線）

- **校準資料**：**Lidar AI Solution / libspconv** 可在 C++ 路徑處理大 N；**本 repo 的 spconv FX PTQ（`prepare_fx` → observers）在 GPU 上對 N 是 ~O(N²) 記憶體**，全場景幾乎必炸，與「校準 sample 數」無關。因此 **AWML 預設對每幀做 voxel 子採樣**（deploy / env / 預設 4096），並提供 **`bevfusion/quantization/quantize.py --spconv-calib-max-voxels`** 覆寫。Deploy 的 `quantization` 請用 **`deploy_cfg.get("quantization")`** 讀取（舊版僅 `getattr` 可能讀不到 MMEngine `Config` 裡的鍵，導致 cap 未生效而 OOM）。
- **ONNX 匯出**：長期應逐步靠攏 **Lidar AI Solution** 形態——**稀疏專用 ONNX + libspconv（或同等 plugin）**、稠密段標準 ONNX/QDQ；目前 AWML 仍以 **torch.onnx.export + FP32 sparse shadow** 等過渡手段為主，見進度 README。

---

## 一、三句話結論（先看這裡）

1. **`torch.ao.quantization` + `convert_fx` 後的 spconv 稀疏塔** 圖裡會出現 **`qint8`、`aten::_empty_affine_quantized`** 等；**標準 `torch.onnx.export` 無法把「整段」原封不動變成通用 ONNX**（你會遇到 `UnsupportedOperatorError`）。這不是「opset 不夠新」而已，而是 **ONNX 本來就不承載這套 PyTorch 原生量化張量語意**。

2. **業界常見作法**是 **拆段**：  
   - **稀疏塔**：用 **專用 runtime**（**libspconv / TensorRT plugin / 3DSparseConvolution 那類 ONNX 描述 + `.so` 執行**）跑 INT8 或 FP16；  
   - **稠密塔**（SECOND、FPN、head）：用 **一般 ONNX → TensorRT**（或帶 QDQ 的 ONNX）。  
   **Lidar AI Solution** 的 `bevfusion.scn.xyz.onnx` + **libspconv** 就是「**稀疏一段 ONNX + 動態連結庫**」，不是「一顆大 ONNX 裡全是標準 Conv」。

3. **`spconv gencode` 產生的 libspconv 已內建 INT8 kernel**；**不必為了「能跑 INT8」去改 spconv 原始碼**。你要做的是：**從 PyTorch PTQ 模型把 int8 權重與 scale 依 [TENSORRT_INT8_GUIDE](https://github.com/traveller59/spconv/blob/master/docs/TENSORRT_INT8_GUIDE.md) 換算後餵給 C++/plugin**。**libspconv 不會自動讀你的 `.pth`**。

---

## 二、名詞對照（白話）

| 名詞 | 意思 |
|------|------|
| **稀疏塔** | BEVFusion 的 `pts_middle_encoder`（voxel → spconv 3D → BEV feature） |
| **稠密塔** | `pts_backbone`、`pts_neck`、`bbox_head` 等一般 2D Conv / Transformer |
| **PTQ INT8（spconv）** | `prepare_fx` → 校準 → `convert_fx` → spconv 的 `transform_qdq` 等，得到 **GraphModule + qint8** |
| **ONNX** | 多數引擎認得的 **標準算子清單**；**不含** PyTorch 私有量化張量算子 |
| **TensorRT** | NVIDIA 推理引擎；**自訂稀疏層**要靠 **plugin** 或 **段外執行** |
| **gencode / libspconv** | `python -m spconv.gencode ...` 產出 **純 C++ 庫**，內含 **FP32/FP16/INT8** 等 kernel（見 spconv `gencode/__main__.py` 註解 *keep all int8 kernels*） |

---

## 三、為什麼「整網單一 ONNX」在 PTQ + spconv INT8 下特別難？

- **FP32 全網**：稀疏塔仍是 **特殊實作**，但較少 **`_empty_affine_quantized`**；有時仍會因 **spconv / JIT / 動態 shape** 踩雷，但和 INT8 相比門檻較低。
- **PTQ + `convert_fx` 稀疏塔**：圖中大量 **量化原語**；ONNX exporter **沒有**對應 symbolic → **匯出失敗**或圖無效。
- **稠密端若用 pytorch_quantization**：`torch.onnx.export` 牽涉 **fake quant / QDQ**；需 **`TensorQuantizer.use_fb_fake_quant`** 與 **`quant_conv.py`** 在 trace 時仍走 quantizer（見進度 README §5.7）。

因此：**「正確」若定義為「一檔 `.onnx` 用 `trtexec` 直接吃滿 BEVFusion + spconv 真 INT8」—— 實務上 **通常不成立**；應改問 **「哪一段用 ONNX/TRT，哪一段用 libspconv / PyTorch」**。

---

## 四、建議部署形態（由易到難）

### 路線 0：僅驗證精度／線上 PyTorch — **不強求 ONNX**

- **PyTorch + PTQ checkpoint + GraphModule 稀疏塔**：與 spconv [INT8_GUIDE](https://github.com/traveller59/spconv/blob/master/docs/INT8_GUIDE.md) 一致。  
- **不需** gencode；**不需**改 spconv（AWML 用 **runtime patch** `spconv_quantized_add_patch.py` 即可，見實作歷程）。

### 路線 1：**與 Lidar AI Solution 類似 — 多段產物（強烈建議當作目標架構）**

概念上對齊 **Lidar_AI_Solution / 3DSparseConvolution** 文件中的用法：

- **稀疏子網**：匯出 **僅描述稀疏部分的 ONNX**（例如 `*.scn*.onnx`），實際 **INT8/FP16 推理由 libspconv（或同等 `.so`）執行**，ONNX 常作 **圖結構／權重載入協議**，**不是** 純 TensorRT 內建節點拼滿。
- **稠密子網**：**另一個（或多個）ONNX**，內容為 **FP32/FP16 或帶 QDQ 的 Conv**，可用 **`trtexec` / `build_engine` from ONNX**。

**你是否應該拆成多個 ONNX？**  
- **若目標是 TensorRT + 真 spconv INT8**：**是**，思維上應 **至少**把 **「稀疏」與「稠密」拆開**；稀疏段接 **plugin / libspconv**，與 **單檔標準 ONNX** 分工。  
- 這與 **Lidar AI Solution** 的 **BEVFusion：`bevfusion.scn.xyz.onnx` + voxels/coors 輸入 + libspconv** 描述一致（見其 `libraries/3DSparseConvolution/README.md`）。

### 路線 2：**單一 ONNX 僅涵蓋「可匯出」子圖**

- 例如：**ONNX 從「稀疏塔輸出之後」切開**（輸入為已對齊的 BEV feature `[B,256,H,W]`），前面稀疏 **留在 PyTorch 或 C++**。  
- 稀疏 **FP32** 較容易維持幾何一致；稀疏 **INT8** 仍建議 **路線 1**，不要硬塞進標準 ONNX。

### 路線 3：**spconv 官方 PyTorch → TensorRT 路線**

- 文件：**[docs/TENSORRT_INT8_GUIDE.md](https://github.com/traveller59/spconv/blob/master/docs/TENSORRT_INT8_GUIDE.md)**  
- 範例：**`example/mnist/mnist_ptq.py`**（PTQ）、**`example/mnist/mnist_net_transform.py`**（`NetworkInterpreter` 把 FX 圖接到 **TensorRT Python API**）。  
- 這是 **「自己寫翻譯器」** 的正式參考，**不是** 一鍵工具；BEVFusion 需 **依樣擴充 node handler**。

---

## 五、BEVFusion 要怎麼實作，才「比較正確」朝向 ONNX + TRT + spconv INT8？

下面以 **可執行順序** 寫（需團隊分工 C++/TRT 時）：

1. **PTQ 真值在 PyTorch**  
   - 沿用 AWML：`bevfusion/quantization/quantize.py` + `deploy_config_int8.py`，稀疏 **`basicblock_fx`**、`convert_fx` 後 **數值對齊**。  

2. **定義切點**  
   - **切點 A**：voxels + coors → **稀疏塔輸出**（BEV `float` 或你們協定的格式）。  
   - **切點 B**：BEV → **head 輸出**。  

3. **稀疏段**  
   - **選項 A**：**PyTorch INT8** 跑稀疏（無 ONNX），後接 TRT。  
   - **選項 B**：**導出稀疏專用 ONNX + libspconv**（Lidar 路線），從 PTQ 模型 **抽權重／scale** 餵給 runtime（見第七節）。  
   - **選項 C**：**NetworkInterpreter** 建 TRT + plugin（spconv MNIST 範例擴充）。  

4. **稠密段**  
   - **FP32/FP16 ONNX → TRT**；若要保持 **QDQ**，需對 **pytorch_quantization** 與 exporter 對齊（AWML 已部分處理 trace 時 fake quant）。  

5. **幾何對齊**  
   - `bbox_head.bev_pos` 與 **FPN 輸出空間尺寸** 必須一致；INT8/FX 若讓 `dense()` 解析度錯亂，需 **pool 回 `grid_size // out_size_factor`**（見 AWML `bevfusion.py` `_align_lidar_bev_to_head_grid` 與 `sparse_encoder` 相關 workaround）。  

---

## 六、`spconv gencode` 要不要重新產？要不要改 spconv 原始碼？

| 問題 | 建議答案 |
|------|----------|
| **gencode 是否「本來就支援 INT8」？ | **是**，產生的 lib 會帶 **int8 inference** 相關 kernel（見 spconv 原始碼註解）。 |
| **每次 PTQ 都要 regen？ | **否**。PTQ 改的是 **權重與 scale**，不是 **spconv 版本 / CUDA arch / 算子集合** 時，**不必**為 PTQ 重跑 gencode。 |
| **何時要 regen？ | 升級 **spconv/cumm**、改 **CUDA arch 列表**、從 `inference_only=False` 改 True 等 **建置選項** 變更時。 |
| **要不要改 spconv 原始碼？ | **部署 INT8 推理**：**不必**（pip 版即可）。AWML 用 **`spconv_quantized_add_patch.py` runtime patch**，避免改 fork。 |
| **若要走 libspconv C++？ | 你需要的是 **與該 `.so` ABI 相容的產物** + **自寫載入 PTQ 權重的程式**；不是改 spconv 核心邏輯才能「支援 INT8」。 |

---

## 七、怎麼讓 libspconv「吃到」PTQ 的 spconv INT8 層資訊？

**libspconv 不讀 `.pth`，也不自動理解 GraphModule。** 你要自己做 **「匯出適配層」**：

1. 在 **PyTorch** 裡對 **每個量化稀疏層**（型別見 spconv `TENSORRT_INT8_GUIDE` 的 `isinstance` 分支）取出：  
   - **int8 權重**（如 `q_weight.int_repr()`）  
   - **`q_per_channel_scales()`**  
   - **FP32 bias**  
   - **該層 input/output activation scale**（與 FX 圖上 Q/DQ 一致）  

2. 用官方公式換算 **給 implicit gemm 的**  
   `scale_for_spconv_implicit_gemm`、`bias_for_spconv_implicit_gemm`（見 **同一文件** 的 Python 區塊）。  

3. 在 **C++ / plugin** 裡 **cudaMemcpy** 到裝置，呼叫 **`ConvGemmOps::implicit_gemm`**（或你們包好的 API）。  

4. **與 Lidar 生態對齊**：若使用 **他們的 ONNX + parser**，通常還要滿足其 **靜態 shape、voxel 數上限** 等約定（見 3DSparseConvolution README）。

---

## 八、和「整網 `torch.onnx.export`」相比，你該選哪條？

| 目標 | 較務實做法 |
|------|------------|
| 快速在伺服器上跑 **PTQ INT8** | **路線 0**：PyTorch，不匯整網 ONNX。 |
| 車載 / TRT 延遲 | **路線 1 或 3**：**拆段** + **稀疏用 libspconv 或 plugin** + **稠密 ONNX→TRT**。 |
| 堅持 **單一 ONNX 檔** | 僅當 **稀疏已改為 FP32 可匯出子圖** 或 **稀疏不在 ONNX 內**（ONNX 從 BEV 開始）時較可行；**含真 qint8 spconv 的單檔通用 ONNX 不建議當目標**。 |

---

## 九、參考文件與範例（spconv 官方）

- [docs/INT8_GUIDE.md](https://github.com/traveller59/spconv/blob/master/docs/INT8_GUIDE.md) — PyTorch 內 PTQ/QAT / FX 注意事項  
- [docs/TENSORRT_INT8_GUIDE.md](https://github.com/traveller59/spconv/blob/master/docs/TENSORRT_INT8_GUIDE.md) — scale/bias、plugin、TRT 版本、`record_voxel_count`  
- [docs/PURE_CPP_BUILD.md](https://github.com/traveller59/spconv/blob/master/docs/PURE_CPP_BUILD.md) — `gencode` → libspconv  
- `example/mnist/mnist_ptq.py`、`mnist_net_transform.py`、`custom_fx2trt.py` — **MNIST 尺度**的 PTQ 與 FX→TRT 示範  

---

## 十-A、AWML 已內建「路線 1」拆段匯出（sparse + dense）

當 `deploy_cfg["components"]` 同時包含 **`bevfusion_sparse`** 與 **`bevfusion_dense`**（且**不**再使用 `bevfusion_main_body`）時：

1. **ONNX**：`BEVFusionONNXExportPipeline` 會寫入同一個 `onnx/` 目錄下的  
   - `bevfusion_sparse.onnx`：voxels / coors / num_points → `lidar_bev`  
   - `bevfusion_dense.onnx`：`lidar_bev` → `bbox_pred` / `score` / `label_pred`（含與單檔相同的 TopK 修復）  
   - **PTQ 且 `pts_middle_encoder` 為 `convert_fx` GraphModule 時**：匯出前會 **暫時**換成重建的 **FP32 融合稀疏塔** 再 `torch.onnx.export`，結束後還原 GraphModule；稀疏 ONNX 為 **浮點圖**（對齊 Lidar `*.scn.onnx` + libspconv 以 FP16/FP 解析的思路），**不等於** PyTorch 內真 INT8 稀疏推理。
2. **TensorRT**：`BEVFusionTensorRTExportPipeline` 會掃描目錄內多個 `.onnx`，依各 component 的 `engine_file` 產生 **兩顆 engine**。
3. **推理**：`BEVFusionONNXPipeline` / `BEVFusionTensorRTPipeline` 會自動 **先跑稀疏再跑稠密**（無須改 CLI 子命令）。

**範例設定檔**：
  - FP / 未開量化：[`config/deploy_config_split.py`](./config/deploy_config_split.py)
  - **已 PTQ 的 checkpoint**（spconv INT8 + 稠密 QAT）：[`config/deploy_config_split_int8.py`](./config/deploy_config_split_int8.py)

```bash
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config_split.py \
  /path/to/your_model_cfg.py

# PTQ checkpoint + 拆段匯出
python -m deployment.cli.main bevfusion \
  deployment/projects/bevfusion/config/deploy_config_split_int8.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_4xb8_j6gen2_base_120m_fx.py
```

**限制**：目前僅支援 **純 LiDAR**（`fusion_layer is None` 且 `img_backbone is None`）；否則匯出會明確報錯。  
**稠密段 TRT profile**：請依 checkpoint 調整 `bevfusion_dense.tensorrt_profile.lidar_bev` 的 **C / H / W**（預設假設 `256×180×180` 量級）。

**常見 TRT 錯誤**：`Dimension mismatch for tensor lidar_bev ... axis 1, profile has min=64 ... but tensor has 256` — ONNX 裡 **C 維是靜態的**，profile 的 **min/opt/max 在 axis 1 必須都等於該 C**（例如皆為 `256`），不能只對 channel 做 64–512 區間；僅 **H、W** 適合用 min/opt/max 拉開。

---

## 十、AWML 內相關檔案（方便跳轉）

| 檔案 | 用途 |
|------|------|
| `deployment/projects/bevfusion/export/onnx_export_pipeline.py` | 單檔或 **拆段** ONNX 匯出 |
| `deployment/projects/bevfusion/config/deploy_config_split.py` | **路線 1** 雙 component（FP） |
| `deployment/projects/bevfusion/config/deploy_config_split_int8.py` | **路線 1** + PTQ checkpoint / spconv INT8 |
| `deployment/projects/bevfusion/quantization/spconv_quantized_add_patch.py` | **不改 spconv 安裝檔**的 runtime patch |
| `deployment/projects/bevfusion/quantization/spconv_int8.py` | prepare_fx / convert_fx 流程 |
| `deployment/projects/bevfusion/io/model_loader.py` | PTQ 載入、結構對齊 |
| `projects/BEVFusion/bevfusion/bevfusion.py` | BEV 特徵與 head 網格對齊等 |
| `projects/BEVFusion/bevfusion/sparse_encoder.py` | `_conv_out_to_bev`、Z 維 workaround |

---

## 十一、總結表：你的問題直接回答

| 問題 | 回答 |
|------|------|
| BEVFusion 怎樣才「正確」 toward `.onnx` + TRT + spconv INT8？ | **拆段**：稀疏 **libspconv / plugin / 專用 ONNX 協議**；稠密 **標準 ONNX→TRT**；PTQ 數值在 PyTorch 對齊後再 **抽權重給 C++**。 |
| gencode 要重產嗎？ | **不必為每次 PTQ**；**版本/arch/算子集合變了再產**。 |
| 要改 spconv 嗎？ | **不必**；AWML 用 **patch**；libspconv 用 **官方公式餵 scale/weight**。 |
| libspconv 怎麼吃 PTQ？ | **自寫 dump + C++ 載入**，對齊 **TENSORRT_INT8_GUIDE**。 |
| 要像 Lidar AI 拆多 ONNX 嗎？ | **若目標是 TRT + 真稀疏 INT8，思維上應拆**；與其 **單檔全標準 ONNX** 務實。 |

---

*本文件為架構說明；實作細節與 commit 對照仍以 `4_spconv_int8_implementation_history_zh.md` 為準。*

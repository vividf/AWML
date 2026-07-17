# StreamPETR 部署遷移說明（中文）

本文件說明 StreamPETR 從舊部署程式碼遷移到新 `deployment/` 框架的內容（對應
`spec_streampetr_migration.md` 的 Phase 0–2），以及「與 reference 比對」的具體方法與限制。
英文版專案說明見 [README.md](README.md)。

---

## 1. 做了什麼

### 1.1 背景：舊部署 vs 新框架

舊的 StreamPETR 部署只有兩個檔案（`projects/StreamPETR/deploy/torch2onnx.py` +
`containers.py`，共 389 行），行為是：

- 跑三次 CLI，各自把模型的一段導出成一個 ONNX：
  `extract_img_feat`（影像 backbone + neck）、`position_embedding`（3D 位置編碼）、
  `pts_head_memory`（decoder head + 時序記憶佇列）
- **所有輸入都是 `np.random` 假資料**，元件之間沒有真正串接
- 沒有 data loader、沒有 inference、沒有驗證、沒有評估

新框架要求每個專案是一個完整 bundle（config / io / export / inference / evaluation /
runner / entrypoint），掛進共用的 CLI 與 orchestrator。本次遷移把 StreamPETR 做成
`deployment/projects/streampetr/`，目前完成 **ONNX 導出鏈**（Phase 0–2）。

### 1.2 新增的檔案與角色

| 檔案 | 角色 |
| --- | --- |
| `__init__.py` | 向框架 registry 註冊 `streampetr`，CLI 自動出現子命令 |
| `entrypoint.py` | 自組線路的 `run(args)`。不能用共用的 `run_detection3d_deployment`，因為它寫死 LiDAR 的 `PointCloudDataLoader`，StreamPETR 是相機模型 |
| `config/deploy_config.py` | 部署設定：3 個元件的 I/O 名稱（**凍結契約**）、opset 18、onnxsim、TRT profile（Phase 3 用）、eval/verify 停用 |
| `config/streampetr_deployment_config.py` | 型別化設定：驗證 3 個必要元件存在；**強制 `num_warmup=0`**（warmup 會重放樣本、污染時序記憶佇列） |
| `io/model_loader.py` | 建模型 + 載 checkpoint；把 decoder 兩層 attention 換成可導出的 `PETRMultiheadAttention`（flash-attn 無 ONNX 路徑，沿用舊 code 的手術） |
| `io/data_loader.py` | 多視角相機 loader。`StreamPETRDataset` 依 (scene, timestamp) 排序，**index 順序即時序順序**；另提供 `is_sequence_start` 旗標 |
| `io/sample_types.py` | `StreamPETRExportSample`：導出用的型別化樣本 |
| `export/onnx_models/` | 三個 tracing 容器，從 `containers.py` **逐字移植**（維持部署圖形契約） |
| `export/sample_extractor.py` | **與舊版最大的差異**：載入一張真實影格，依序跑 encoder → position embedding，把上游輸出接給下游當 tracing 輸入；記憶佇列用零初始化（clip 開始時的真實狀態） |
| `export/component_builder.py` | 把模型切成 3 個 `ExportableComponent`；導出前設 `with_dn=False`（denoising 只在訓練用） |
| `runner.py` | 薄薄的 runner：載模型 + 把 extractor/builder 接進共用 `OnnxExportPipeline` |
| `evaluation/executor.py` | Backend executor；`create_pipeline` 目前拋 `NotImplementedError`（推論 pipeline 是 Phase 4） |

### 1.3 順手修掉的問題（不在原計畫內，但被 e2e 逼出來）

1. **`attention.py` 頂層 import flash_attn**：部署 container 沒裝。改成 optional import，
   只有真的建構 `FlashAttention` 才報錯。訓練環境行為不變。
2. **`backbones/__init__.py` import EVAViT**：拉進 fvcore/timm 等訓練專用依賴。改成
   optional（T4 模型用 VoVNetCP，不受影響）。
3. **`Petr3D.train()` 覆寫沒有 `return self`**：導致 `model.float().cpu().eval()` 鏈式呼叫
   最後拿到 `None`。這也解釋了舊 exporter 為何逐句呼叫 `tm.float(); tm.cpu(); tm.eval()`。
4. **dataset 的 `prev_exists` 在 index 0 有負索引 wrap-around**：`flag[index-1]` 在 index 0
   會拿到「最後一筆」的 flag，單一 scene 的 test set 會誤判第 0 帧不是序列起點。
   在 loader 端強制 index 0 為序列起點。
5. **intrinsics 3×3 vs 4×4**：dataset 給 3×3 `cam2img`，部署契約是 4×4。extractor 補成
   homogeneous 4×4（head 只讀 `[...,0,0]`/`[...,1,1]` 兩個焦距，數值等價）。

### 1.4 執行方式

```bash
# 在部署 container 內、repo 根目錄執行
python -m deployment.cli.main streampetr \
    deployment/projects/streampetr/config/deploy_config.py
# 產物：work_dirs/streampetr_deployment/onnx/{extract_img_feat,position_embedding,pts_head_memory}.onnx
```

---

## 2. 「與 reference 比對」的細節

### 2.1 比對基準（oracle）是什麼

`work_dirs/streampetr/` 內的三個檔案，即 **實際部署在線上的 v2.5 模型 ONNX**
（model-zoo 發佈版，由舊 `torch2onnx.py` + onnxsim 產生）：

```
simplify_extract_img_feat.onnx
simplify_position_embedding.onnx
simplify_pts_head_memory.onnx
```

搭配同目錄的 v2.5 checkpoint（`best_NuScenesmetric_T4Metric_mAP_epoch_34.pth`）與
dump 出來的完整訓練 config。新導出使用**同一顆 checkpoint**，因此差異只會來自導出機制本身。

### 2.2 比了哪些東西

對每一對（新導出 vs reference）做三個維度的結構比對：

1. **Graph inputs**：張量「名稱、dtype、完整靜態 shape」的**有序完全比對**。
   例如 `pts_head_memory` 的 10 個輸入
   （`x[1,5,256,30,40]`、`pos_embed[1,6000,256]`、…、`pre_memory_egopose[1,1024,4,4]`）
   逐一吻合，順序也一致。
2. **Graph outputs**：同上，14 個輸出逐一吻合，包含容易出錯的細節——
   `post_memory_timestamp` 是 **DOUBLE**（f64 zeros 與 f32 記憶 cat 後的型別提升），
   `post_memory_*` 長度是 **1280**（memory_len 1024 + top-k 256，host 端要切回 `[:1024]`）。
3. **Op-count 分布**：統計整張圖每種 ONNX op（Conv、MatMul、TopK…）的出現次數，
   與 reference 逐項比較。三個元件的 op 分布**全部零差異**。這代表 torch 2.8 重現出的
   圖形結構與當年導出的部署版本等價，不只是介面對得上。

比對腳本核心邏輯（可直接重跑）：

```python
import onnx

def io_sig(path):
    m = onnx.load(path, load_external_data=False)
    def fmt(vs):
        return [(v.name,
                 onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                 tuple(d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim))
                for v in vs]
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return fmt(m.graph.input), fmt(m.graph.output), ops

# 對每一對 (新導出, reference) 比較 inputs / outputs / op-count 三者是否完全相等
```

### 2.3 一個重要細節：onnxsim 的剪枝行為也要重現

舊 code 傳給 `torch.onnx.export` 的輸入，和最終部署圖的輸入**不一樣**：

- `position_embedding` tracing 時有 4 個輸入（含 `img_feats`），但圖中只用到它的
  shape（trace 時固化成常數），onnxsim 把沒用到的 `img_feats` 輸入剪掉 → 部署版只有 3 個輸入。
- `pts_head_memory` tracing 時有 11 個輸入（含 `data_timestamp`），但 timestamp 運算
  刻意放在 TensorRT 之外（原始碼中該行被註解），onnxsim 剪掉 → 部署版只有 10 個輸入。

新導出沿用「trace 全部輸入 → onnxsim 剪枝」同一條路徑，最終圖形與 reference 的輸入數
一致（3 / 10），證明這個微妙行為被完整重現。第一輪比對抓到的唯一差異就是
intrinsics 3×3 vs 4×4（§1.3-5），修正後三個元件全數 `MATCH`。

### 2.4 這個比對「沒有」涵蓋什麼（誠實的邊界）

- **沒有比對權重數值**：兩邊用同一顆 checkpoint，且 op 結構一致，但沒有逐一 diff
  initializer 的位元內容。
- **沒有跑數值等價驗證**：即「同一輸入 → 兩張圖輸出誤差在容忍範圍內」。這是
  migration spec 的 **Phase 5**（cross-backend verification），需要先有推論 pipeline
  （Phase 4），且時序模型必須從 clip 起點、以相同的零初始化記憶佇列逐帧比對。
- **沒有比對 TensorRT engine**：Phase 3。

換句話說，目前的結論是「**導出機制遷移後，產出的圖形在結構與介面契約上與部署版本等
價**」；數值層級的等價驗證由後續 Phase 完成。

---

## 3. 目前狀態與後續

- [x] Phase 0–2：scaffolding、io、ONNX 導出（Docker e2e 通過，I/O＋op 分布與部署參考一致）
- [x] Phase 3：TensorRT 導出（FP16）。注意：engine build 會同時榨乾 GPU/CPU，散熱受限的
      筆電會觸發 ACPI 過熱保護關機——build 前先鎖時脈/限功耗
- [x] Phase 4–5：有狀態推論 pipeline（三後端）＋跨後端數值驗證（從 clip 起點、零記憶）
- [x] Phase 6：評估（5 幀 smoke run）——三後端 mAP 一致（BEV-center 0.544–0.549），
      TensorRT FP16 延遲 65.5ms ≈ PyTorch 的 3.6 倍快

實作過程逼出的三個 pipeline 修正（值得記住）：
1. decoder 的 query 數是 `num_query + num_propagated`（644+256=900），不是 config 的
   `num_query`——參考 ONNX 的 `outs_dec [6,1,900,256]` 就是證據
2. `NMSFreeCoder.decode_single` 內部用 `.view(-1)`，ONNX/TRT 輸出經 `torch.from_numpy`
   進來後 transpose/reshape 非連續 → decode 前需 `.contiguous()`
3. 記憶佇列的 timestamp 運算在 host 端（graph 內被註解）：餵入前 `+t`、收回後 `-t`
   （條目語義＝「相對當前幀的年齡」），並把 `post_memory_*[:, :1024]` 切片回饋

評估 metrics 已對齊訓練設定：
`projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline_t4metric_v2.py`
（比照 CenterPoint 的 `*_t4metric_v2.py` 慣例：繼承訓練 config、只換 evaluator；
51.2m 評估範圍對應 v1 的 `eval_class_range`）。此 variant 同時**釘住 v2.5 artifact 的
5 類配置**——共用 t4dataset base 後來擴成 7 類（+traffic_cone/+barrier），7 類 head 會
靜默載不進 5 類 checkpoint 的分類分支，mAP 直接崩到 0.05（已實測驗證，釘回 5 類後
數字逐位恢復）。全 19 幀本地 clip：PyTorch 0.6320 / ONNX 0.6305 / TRT-FP16 0.6366 mAP
（BEV-center），TRT 延遲 56.7ms。注意絕對值是「單一 clip、19 幀」的局部數字（該 clip
行人 GT 為 0），不能與 model-zoo 的 8,453 幀全量評估（mAP 0.45）直接比較——跨後端
一致性才是這裡的判準。

設計依據與各項決策的取捨，見 repo 根目錄的 `spec_streampetr_migration.md`。

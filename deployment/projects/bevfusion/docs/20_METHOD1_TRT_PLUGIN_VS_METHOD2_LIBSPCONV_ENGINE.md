# 稀疏塔 INT8：`ImplicitGemmInt8` 管線（部署流程、方法對照與實作現況）

本文件對齊 **`10`～`19`** 與**目前 AWML BEVFusion deployment 程式碼**（`deploy_config_split_int8.py`、`sparse_int8_onnx_transform.py`、`export.mode` 等）。  
先前版本以「純策略分析」為主；本版**補上實際 CLI 流程**、**削減已主線落地的重複描述**，並對**尚未完成或需加強**的項目標註**難度**與**實作要點**。

---

## 術語：「Path B」是什麼？較好的名字

歷史上（見 **11**、**10**）文件用 **Path A / Path B** 區分兩種稀疏 INT8 後端：

| 舊稱 | 含義 |
|------|------|
| **Path A** | **預編譯 `libspconv.so`**（NVIDIA Lidar AI Solution 風格）：稀疏塔在 **TensorRT 外**用 **自訂 ONNX + libspconv 解析**，與稠密 TRT 在 BEV 交界。 |
| **Path B** | **仍在 TensorRT 圖內**：標準 **`autoware::ImplicitGemm` ONNX**，經 **`sparse_int8_onnx_transform`** 換成 **`ImplicitGemmInt8`**，由 **`libimplicit_gemm_int8_plugin.so`**（cumm INT8 kernel）執行。 |

若團隊已**不再維護 Path A 作為日常選項**，「Path B」就不再表達「第二條路」，而只是**口頭遺留詞**，新建議命名如下（新文件優先用 **描述性名稱**，舊 grep / 檔名可保留 Path B 作別名）：

| 建議稱呼 | 說明 |
|----------|------|
| **ImplicitGemmInt8 管線**（推薦） | 直指 ONNX op + TRT plugin 與腳本 `sparse_int8_onnx_transform`，一聽就知道技術棧。 |
| **TensorRT 稀疏 INT8 外掛管線** | 強調稀疏塔仍在 TRT、靠 **custom plugin**，而非 libspconv 獨立 runtime。 |
| 英文短寫 | **TRT ImplicitGemm INT8 stack** 或 **sparse encoder INT8 (TRT plugin)**。 |

舊文件中的 **「Path B」** ≈ 本文件的 **「ImplicitGemmInt8 管線」**。

---

| 序號 | 文件 | 用途（摘要） |
|------|------|--------------|
| 10 | `10_int8_trt_gap_analysis.md` | 鴻溝、libspconv 獨立運行時 vs TRT |
| 11 | `11_int8_pathb_autoware_plugin.md` | ImplicitGemmInt8 plugin、Q/DQ、transform（檔名仍為 pathb） |
| 12 | `12_int8_sparse_pipeline_ptq_onnx_trt.md` | PTQ → ONNX → TRT 全鏈 |
| 13–19 | 13–19 | 評測、split 除錯、加速計畫、profile、層級比較 |

---

## 0. ImplicitGemmInt8 管線：標準操作（與 repo 一致；路徑以 `deploy_config` 為準）

`deployment/configs/enums.py` 中 **`export.mode`** 合法值為：**`onnx` | `trt` | `both` | `none`**（沒有字串 `tensorrt`）。  
`deploy_config_split_int8.py` 內 **`export.work_dir`** 預設常為 `work_dirs/bevfusion_split_int8_deployment_*`；以下路徑請改為**你實際設定的 `work_dir`**；產物目錄為 `{work_dir}/onnx/`、`{work_dir}/tensorrt/`。

| 步驟 | 目的 | 指令 / 設定要點 |
|------|------|-----------------|
| **0（可選）** | 編譯 `ImplicitGemmInt8` plugin | `deployment/projects/bevfusion/cpp/int8_plugin` CMake 產出 `libimplicit_gemm_int8_plugin.so`；在 `tensorrt_config.plugin_libraries` 與 Autoware `.so` 並列（見 **11**、**16**） |
| **1** | PTQ，產含 `_amax` 的 `.pth` | `python deployment/quantization/bevfusion_quantization.py ptq ... --deploy-cfg deployment/projects/bevfusion/config/deploy_config_split_int8.py ...`；若只要稀疏塔校準、與 **Preset C** 一致，建議加 **`--sparse-int8-only`**（見 deploy 檔頭註解） |
| **2** | 只匯 ONNX（FP16 `ImplicitGemm`） | 在 deploy 設 **`export.mode = "onnx"`**（或 **`"both"`** 會同時建 TRT，一般不用于「先 transform 再建 engine」）後執行：`python -m deployment.cli.main bevfusion <deploy_cfg> <mmconfig>` |
| **3** | 保留乾淨 FP16 稀疏 ONNX 備份 | 例如：`mv .../onnx/bevfusion_sparse.onnx .../bevfusion_sparse_fp16.onnx`（避免覆寫後無法重跑 transform） |
| **4** | ONNX 後處理：換成 `ImplicitGemmInt8` + scale | `python -m deployment.projects.bevfusion.export.sparse_int8_onnx_transform --onnx ..._fp16.onnx --checkpoint <同 PTQ> --output .../onnx/bevfusion_sparse.onnx`；**Option2**：加 **`--deploy-cfg`** 讀取 `spconv_int8_fp16_layers`、`spconv_int8_fuse_implicit_gemm_relu` 等 deploy 設定 |
| **5** | 只建 TensorRT、不重匯 ONNX | 設 **`export.mode = "trt"`**，再跑同一 `deployment.cli.main bevfusion ...`；會依現有 ONNX 建 `bevfusion_sparse.engine` / `bevfusion_dense.engine` |

**已在程式層支援、無需你再「發明」的項目**：  
`sparse_int8_onnx_transform`（含 **`spconv_int8_fp16_layers`**、stem 對齊、initializer/`graph.input`、`--audit-report`）、deploy 內 **`spconv_do_sort`**（INT8 建議 `False`，與 New3D 行為對齊）、**NVIDIA shadow scheme A**（無 FX GraphModule 時可由 `resolve_sparse_onnx_shadow` 觸發，見 **12**）、**`cpp/int8_plugin`（ImplicitGemmInt8）** 與 **`tensorrt_config.plugin_libraries`** 雙 plugin 配置。

---

## 1. 問題陳述（方法 1 / 2 仍要解什麼）

- **ImplicitGemmInt8 管線**對外仍多為 **FP16 I/O**（見 **11**），層間 **FP16 邊** → 層內再 **FP16→INT8**，與 **15**、**19** 的量測一致。  
- **TRT 不會**對 **custom plugin** 自動做 Q/DQ fusion（**10**）。  
以下「方法」指**在現狀上再往哪投資**。

---

## 2. 方法 1：深化 ImplicitGemmInt8 管線（已落地 vs 待加強）

### 2.1 已在主線或部分完成（本節不再展開長文）

| 項目 | 狀態 |
|------|------|
| PTQ + `_amax` + `sparse_int8_onnx_transform` | **已主線** |
| `ImplicitGemmInt8` + `plugin_libraries` | **已主線** |
| 乾淨 ONNX（scheme A shadow） | **已實作**（條件不符時仍可能出現 Q/DQ，見 **11 紀錄 8**） |
| 每幀權重量化等 micro-opt（doc **15** §8.1 類） | **已做**，整段 sparse 加速仍受 **pair_gen / memory** 限制 |
| `spconv_do_sort=False` 烘焙進 ONNX | **已可由 deploy 設定** |

### 2.2 待加強項目（難度與實作介紹）

| 項目 | 難度 | 說明與建議實作方向 |
|------|------|-------------------|
| **剝除或合併冗餘 Q/DQ（方案 B）** | ⭐⭐ | 若 shadow 未觸發，圖上仍有 **Q/DQ 獨立 layer**。**實作**：在 `sparse_int8_onnx_transform` 或獨立 ONNX pass 中，對 `ImplicitGemm` 輸入追溯 `DequantizeLinear` → 改接線到 DQ 前浮點張量（**11** 紀錄 8-4 表）。需回歸 TRT build + 單 frame `lidar_bev` 數值。 |
| **Conv / ImplicitGemm 與 ReLU 吸收為單一 plugin 屬性** | ⭐⭐⭐ | ONNX 上 **ReLU 常為獨立節點**，TRT **不會**與 custom op 自動 fuse。**實作**：(1) 匯出／symbolic 層把 `act_type` 寫進下一個 `ImplicitGemm` 的 plugin attr；(2) 或 plugin 內對固定子圖做 pattern merge。需與 spconv `Activation` 枚舉對齊。 |
| **plugin 單次量化後 INT8 activation 在層間傳遞（INT8 tensor 邊）** | ⭐⭐⭐⭐ | 需改 **`supportsFormatCombination`**、ONNX **input/output dtype**、並處理 **TRT 對 2D `[N,C]` INT8 的限制**（**11**）。上游 voxel 與下游 **BEV→dense** 可能仍須 FP16，**無法**保證消滅 **19** 全部「FP16→INT8」列。 |
| **權重 INT8 常駐、僅 build 時量化一次** | ⭐⭐ | 在 **engine 建立或首次推理前**塞縮排後的 INT8 weight buffer，減少 `enqueue` 內 launch。**實作**：plugin `configure`/`initialize` 讀 initializer，或 TRT **weight streaming** 策略；須與 **動態 shape / 多 profile** 相容性驗證。 |

---

## 3. 方法 2：獨立 spconv runtime（Lidar / New3D 式）— 與 repo 現況

### 3.1 參考架構（**10**）

稀疏塔在 **TRT 外**跑 **libspconv / cumm INT8**，與 **稠密 TRT** 在 **BEV FP16** 交界。

### 3.2 AWML 倉庫內已有檔案（≠ 已接入 Autoware 主線）

| 路徑 | 角色 |
|------|------|
| `deployment/projects/bevfusion/export/libspconv_onnx_exporter.py` | 自訂 SparseConvolution ONNX 導出方向 |
| `deployment/projects/bevfusion/cpp/libspconv_trt_bridge.{hpp,cpp}` | 載入 sparse ONNX / engine + dense TRT，配置見檔內 `BridgeConfig` |

**難度**：把上述接成 **產品級管線**（與 `deployment.cli.main`、eval、Docker、版本矩陣）— **⭐⭐⭐⭐**。  
**實作介紹**：以 **split** 為模型—稀疏段改呼叫 **`LibspconvTrtBridge::init` + sparse forward**，輸出 **`lidar_bev`** 再交給既有 dense pipeline（對齊 **14** 的 I/O 名稱與 shape）；Autoware Universe 側需把「單一 TRT 大包」改為 **雙階段或自訂 compositor**，與 **AWML `pipelines/tensorrt.py`** 的 split 邏輯同源但 backend 不同。

---

## 4. 橫向對照（精簡）

| 維度 | 方法 1：ImplicitGemmInt8 / TRT 外掛深化 | 方法 2：獨立 libspconv（或 New3D）運行時 |
|------|---------------------|----------------------------|
| **與現有 AWML** | 同一套 ONNX/TRT/plugin | 需 bridge + 可能替換 sparse engine |
| **FP16↔INT8** | 受 plugin I/O 與 TRT 圖限制 | 較易長鏈 INT8；BEV 仍常 FP16 |
| **開發量** | 增量（§2.2 表） | 大（bridge + 整合 + CI） |

---

## 5. 建議決策順序

1. 用 **16 / 18 / 19** 確認瓶頸是否仍在 **FP16→INT8**、**pair_gen** 或 **GEMM**。  
2. **方法 1（ImplicitGemmInt8 管線）**：優先 **冗餘 Q/DQ 清除**、再評估 **INT8 I/O** 是否值得。  
3. **方法 2**：僅在方法 1 邊際效益不足、且願承擔 **bridge + 多 binary** 時啟動。

---

## 6. 稀疏 INT8 Encoder：延遲拆解與優化上限（實測摘要）

以下整理同一組 profile 條件下的 **bucket 加總**、**layer-wise plugin timing** 與 **Nsight cast/layout**，並對照 **`18`**（端到端與 bucket）、**19**（層級 FP16→INT8）。數字為**一次性量測快照**，換機/GPU/TRT 版本後請重跑。

### 6.1 TensorRT `IProfiler` bucket（稀疏 engine，INT8）

| Component | Time | % |
|-----------|-----:|------:|
| **pair_gen** | 4.78 ms | 39.6% |
| **implicit_gemm_int8** | 4.53 ms | 37.6% |
| **ReLU** | 0.91 ms | 7.6% |
| **other** | 1.63 ms | 13.5% |
| **cast + layout**（約） | ~0.21 ms | ~1.8% |

**若該項「完全消失」的上限節省**：表格時間本身即為單項上限（例如 pair_gen 最多約 **4.78 ms**）。  
**implicit_gemm_int8**：無法靠「刪節點」拿掉；只能靠 **kernel / tiling / 記憶體存取** 調優（見 **18** §3）。

**pair_gen + implicit_gemm_int8** ≈ **4.78 + 4.53 = 9.31 ms**，約 **77.2%**（39.6% + 37.6%）— **仍是絕對主體**。

### 6.2 FP16→INT8 conversion（plugin 內部計時加總）

由 **layer-wise** `fp16_to_int8` / plugin timing **跨層加總**：

| Scope | FP16→INT8 total |
|-------|----------------:|
| **含 conv_input + 全稀疏 conv** | **0.507 ms** |
| **不含 conv_input** | **0.499 ms** |

**解讀**：若把工作流從「**FP16 activation →（層內）量化 → INT8 sparse conv**」改成「**INT8 activation 直入 INT8 sparse conv**」，且中間無其它冗餘，理論上節省上限約 **0.50 ms**（與含不含 conv_input 差異極小）。

### 6.3 ReLU fuse

目前 **ReLU bucket** ≈ **0.91 ms**。若將 ReLU **fuse 進**對應 sparse conv（或後續量化 epilogue），理論上節省上限約 **0.91 ms**。  
實務上可能略低：部分 ReLU 已與鄰近 kernel **overlap**、或子圖**無法**完整 fuse。

### 6.4 Cast + layout（Nsight）vs FP16→INT8（layer-wise）

| 來源 | 數值 | 涵義 |
|------|-----:|------|
| **Nsight** cast + layout bucket | ≈ **0.21 ms** | 僅含被命名規則歸入 cast/layout 的 kernel |
| **Layer-wise** FP16→INT8 加總 | ≈ **0.50 ms** | Plugin **內部**對 FP16→INT8 conversion 的計時 |

兩者**統計範圍不同**，不宜直接相加或互減。**FP16→INT8 saving upper bound** 建議取 **≈ 0.50 ms**；**Nsight-visible cast/layout** 節省上限取 **≈ 0.21 ms**。

### 6.5 綜合「假設性」節省與延遲估算（僅作上限參考）

以下 **New latency** 列為**思想實驗**：假設某優化單獨達成且其它耗時不變；**多列合併並不保證線性相加**（可能重疊、或改變後段 scheduling）。示例 baseline：**約 8.95 ms**（請替換為你當次稀疏 encoder **CUDA event / 一致口徑**總時間）。

| Optimization（單獨假設） | Saving（上限） | New latency 估算 |
|--------------------------|----------------:|------------------:|
| 移除 FP16→INT8 conversion | ~0.50 ms | ~8.45 ms |
| Fuse ReLU | ~0.91 ms | ~8.04 ms |
| 僅移除 cast / layout（Nsight 可見部分） | ~0.21 ms | ~8.74 ms |
| FP16→INT8 + ReLU fuse | ~1.41 ms | ~7.54 ms |
| 上述 + cast/layout | ~1.62 ms | ~7.33 ms |

### 6.6 優化優先級（依本次數字）

| 方向 | Priority | 原因 |
|------|----------|------|
| **pair_gen** | **最高** | 單項最大（~4.78 ms） |
| **implicit_gemm_int8 / kernel tuning** | **最高** | 次大（~4.53 ms），且為硬計算本體 |
| **ReLU fusion** | **高** | 上限約 ~0.91 ms |
| **INT8 activation 鏈（省層內 FP16→INT8）** | **中高** | 上限約 ~0.50 ms |
| **cast / layout 清理** | **中** | Nsight 可見上限約 ~0.21 ms；與 0.50 ms 口徑不同 |

**結論**：即使把 **FP16→INT8、ReLU、cast/layout** 推到理想上限，**pair_gen + implicit_gemm_int8** 仍佔大部分時間；長期仍須 **索引／pair 生成**與 **GEMM kernel** 兩條主線並行。

---

## 7. 相關文件索引（10–19）

見本文件開頭表；profile 步驟以 **16** 為準；bucket 語意與端到端 delta 見 **18** §2–3。

---

*路徑、`export.mode`、`checkpoint_path` 以你使用的 `deploy_config_split_int8.py` 實際內容為準；本文件不鎖定單一 `work_dir` 名稱。§6 數字為示例快照，重跑 profile 後請更新表格。*

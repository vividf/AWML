# StreamPETR 程式碼架構說明

本文件解釋 `projects/StreamPETR/` 的完整程式碼結構:每個模組的職責、model 各部分在做什麼、以及一個 batch 從資料到 loss 的完整流程。使用教學(setup / train / deploy 指令)請看 [README.md](../README.md)。

## 0. 論文核心思想(白話版)

StreamPETR 是 **camera-only 的時序 3D 物件偵測模型**(論文:[StreamPETR, ICCV 2023](https://arxiv.org/abs/2303.11926))。要理解它,先看「利用過去幀資訊」的三條路線:

| 路線 | 代表 | 記憶的是什麼 | 成本 |
|---|---|---|---|
| BEV temporal | BEVFormer、BEVDet4D | 過去的**整張 BEV 特徵圖**,warp 到當前幀 | 貴:每幀都要維護/對齊一張大地圖 |
| Perspective temporal | PETRv2 | 過去幀的**影像特徵**,一起餵進 cross attention | 更貴:token 數翻倍,每幀重算 |
| **Object-centric(本論文)** | StreamPETR | 過去幀**偵測出的物體**(object query) | 幾乎免費:每幀只多 ~1k 條 256 維向量 |

用開車做比喻:你開車時不會記住後照鏡每一幀的完整畫面(BEV/perspective 路線),你記的是一條條筆記——「剛剛左後方有台白車,大概在 10 公尺外,正在加速」。StreamPETR 記的就是這種筆記:每偵測完一幀,把最有把握的 256 個偵測結果(特徵向量、3D 位置、速度、當時的時間與車輛姿態)抄進 **memory queue**;下一幀開始時先讀筆記,把上一幀的偵測直接當作這一幀的「初始猜測」繼續 refine。

因此推論是「串流」的:每幀只跑一次 backbone,時序資訊全靠 query 層級的記憶傳遞,幀率不受歷史長度影響。論文中的關鍵數字(和本 repo config 一致):memory queue 存 **N=4 幀 × K=256 個物體 = 1024 條**(code 的 `memory_len=1024`),每幀傳播 **256 條** propagated query,搭配 **644 條**新 query。

---

## 1. 目錄總覽

```
projects/StreamPETR/
├── configs/                  # 訓練 config(default 底座 + t4dataset / nuscenes 覆寫)
├── deploy/                   # ONNX 匯出:模型拆成 3 個 ONNX
├── stream_petr/
│   ├── models/
│   │   ├── detectors/        # Petr3D:整體 orchestrator
│   │   ├── backbones/        # VoVNet / VoVNetCP / EVAViT
│   │   ├── necks/            # CPFPN
│   │   ├── dense_heads/      # StreamPETRHead(3D)、FocalHead(2D 輔助)
│   │   ├── utils/            # PETR transformer、attention、位置編碼、memory 工具
│   │   └── optimizer/        # NoCacheAmpOptimWrapper(AMP workaround)
│   ├── datasets/
│   │   ├── pipelines/        # StreamPETRDataset、影像/BEV 增強、格式化
│   │   └── samplers/         # GroupStreamingSampler(時序 batch 排程)
│   └── core/
│       ├── bbox/             # Hungarian assigner、NMSFreeCoder、match cost、box 編解碼
│       ├── apis/             # ⚠ mmdet2 時代遺留,現行訓練完全不用
│       └── evaluation/       # ⚠ 同上,legacy
└── docs/                     # 本文件、各 dataset 的 release note
```

訓練入口是共用的 `tools/detection3d/train.py`(mmengine Runner);config 裡的
`custom_imports = ["projects.StreamPETR.stream_petr"]` 會觸發
[stream_petr/__init__.py](../stream_petr/__init__.py) 把所有自訂類別註冊進
mmdet3d / mmdet / mmengine 的 registry,之後 config 中的 `type="..."` 字串才找得到類別。

---

## 2. 端到端資料流(訓練一個 iteration)

```
info pkl (T4dataset)
   │  StreamPETRDataset:依 scene_token 排序、算 flag(scene 分組)、prev_exists
   ▼
train_pipeline(逐幀):
   LoadMultiViewImageFromFiles → LoadAnnotations3D → ObjectNameFilter
   → ResizeCropFlipRotImage(影像 IDA 增強,同步改 intrinsics/lidar2img)
   → GlobalRotScaleTransImage(BEV 空間增強,同步改 ego_pose/extrinsics)
   → PadMultiViewImage → NormalizeMultiviewImage
   → StreamPETRLoadAnnotations2D(把 3D GT 投影成各相機的 2D GT)
   → ObjectRangeFilter → PETRFormatBundle3D(轉 tensor)
   ▼
GroupStreamingSampler:把「整個 scene」分配到 batch 的固定 slot,
   每個 slot 依時間順序逐幀送出 → batch 內同 slot = 同一場景的連續幀
   ▼
Petr3D.forward(mode="loss")
   ├─ VoVNet backbone(+GridMask 增強)→ stage4/stage5 特徵
   ├─ CPFPN neck → 256-ch 特徵圖(stride 16)
   ├─ FocalHead(2D 輔助):對每個 pixel 預測 2D box/centerness
   │     → 取 top-k 特徵 token 給 3D head(focal sampling)
   └─ StreamPETRHead(3D 主 head):
         pre_update_memory(依 prev_exists 清空或搬移 memory)
         → 3D position embedding(把 2D 特徵 lift 成 3D 座標編碼)
         → denoising query 生成(訓練專用)
         → temporal_alignment(把 memory 中的舊 query 對齊到當前幀)
         → PETRTemporalTransformer(6 層 decoder,global cross attention)
         → cls/reg branch 輸出 → post_update_memory(存 top-k 進 memory)
   ▼
Loss:每層 decoder 的 (focal cls + L1 bbox) + DN loss + 2D 輔助 loss
   → NoCacheAmpOptimWrapper(AMP)backward → AdamW step
```

推論時(`mode="predict"`)只支援 batch_size=1,逐幀串流;遇到新的
`scene_token` 會重置 memory([petr3d.py:318-322](../stream_petr/models/detectors/petr3d.py#L318-L322))。

---

## 3. Model 各部分在做什麼

### 3.1 `Petr3D` — 整體 orchestrator
[stream_petr/models/detectors/petr3d.py](../stream_petr/models/detectors/petr3d.py)(registry:`Petr3D`)

繼承 mmdet3d 的 `MVXTwoStageDetector`,但只用 camera 分支。職責:

| 方法 | 做什麼 |
|---|---|
| `extract_img_feat()`([L91](../stream_petr/models/detectors/petr3d.py#L91)) | 把 (B,N,3,H,W) 多相機影像攤平 → GridMask 增強 → backbone → neck,reshape 回 (B,T,N,C,H,W) |
| `forward_train()`([L253](../stream_petr/models/detectors/petr3d.py#L253)) | 多幀輸入時,只有最後 `num_frame_backbone_grads` 幀的 backbone 有梯度,更早的幀用 `no_grad` 跑(只為了填 memory);現行 T4 config 是 `num_frame_losses=1`,即單幀訓練 |
| `obtain_history_memory()`([L124](../stream_petr/models/detectors/petr3d.py#L124)) | 逐幀呼叫 `forward_pts_train`,控制哪幾幀要梯度、哪幾幀要算 loss |
| `forward_roi_head()`([L173](../stream_petr/models/detectors/petr3d.py#L173)) | 跑 2D FocalHead 拿 `topk_indexes`;`aux_2d_only=True` 時推論不跑(省時間) |
| `simple_test_pts()`([L311](../stream_petr/models/detectors/petr3d.py#L311)) | 串流推論:scene_token 變了就 `reset_memory()` 並強制 `prev_exists=0` |
| `_extras_for_t4metric()`([L339](../stream_petr/models/detectors/petr3d.py#L339)) | **T4 特有**:組出 T4Metric 需要的 timestamp / lidar_path / eval_ann_info |
| `train()`([L414](../stream_petr/models/detectors/petr3d.py#L414)) | train↔eval 模式切換時清空 memory(log 裡的 "Cleared memory due to change in mode" 就是這裡) |

注意:`train()` 被 override 且不回傳 self,所以**不要寫 `model.eval()` 的鏈式呼叫**。

同目錄的 `RepDetr3D`([repdetr3d.py](../stream_petr/models/detectors/repdetr3d.py))是多尺度特徵版的變體,現行 config 未使用。

### 3.2 `VoVNet` — 影像 backbone
[stream_petr/models/backbones/vovnet.py](../stream_petr/models/backbones/vovnet.py)(registry:`VoVNet`,config 用 `V-99-eSE`)

VoVNetV2-99:stem(stride 4)+ 4 個 OSA stage。`_OSA_module` 把連續 5 個 3x3 conv 的輸出 dense concat 後用 1x1 conv 聚合,再過 eSE(channel attention)。config 取 `out_features=("stage4","stage5")`(768/1024 ch),`norm_eval=True` 凍結 BN 統計量(fine-tune 慣例)。

- [vovnetcp.py](../stream_petr/models/backbones/vovnetcp.py) 的 `VoVNetCP`:同架構 + gradient checkpointing 省顯存(目前 config 用的是普通 `VoVNet`)。
- [eva_vit.py](../stream_petr/models/backbones/eva_vit.py) 的 `EVAViT`:EVA-02 ViT 大 backbone 備選,T4 config 未使用;`flash_attn` 套件不存在時會退化(已做 optional import)。

### 3.3 `CPFPN` — neck
[stream_petr/models/necks/cp_fpn.py](../stream_petr/models/necks/cp_fpn.py)(registry:`CPFPN`)

標準 FPN 的精簡版:只有 level 0 有 3x3 fpn_conv,其他 level 直接輸出 lateral 1x1 的結果——移除用不到的參數以配合 checkpointing / DDP。輸入 stage4/stage5,輸出 2 個 256-ch 特徵圖;3D head 實際只用 `position_level=0` 那張(stride 16)。

### 3.4 `FocalHead` — 2D 輔助 head(Focal-PETR 的 focal sampling)
[stream_petr/models/dense_heads/focal_head.py](../stream_petr/models/dense_heads/focal_head.py)(registry:`mmdet.FocalHead`)

對特徵圖每個 pixel 用 conv 預測:類別分數、centerness、ltrb 2D box、2D 中心偏移([forward,L196](../stream_petr/models/dense_heads/focal_head.py#L196))。它有兩個作用:

1. **輔助監督**:2D 偵測 loss(QualityFocal + GaussianFocal centerness + L1 + GIoU + centers2d L1)讓影像特徵學得更快——收斂加速器,推論時不需要。
2. **focal sampling**:`cls_score × centerness` 當作重要性權重,取 top-k token 的 `topk_indexes` 交給 3D head,讓 cross attention 只看「可能有物體」的特徵(`train_ratio`/`infer_ratio` 控制取樣比例,目前 config 是 1.0 = 全部,`topk_indexes` 只起排序作用)。

GT 來源:[StreamPETRLoadAnnotations2D](../stream_petr/datasets/pipelines/loading.py) 把 3D GT box 投影到各相機平面即時生成(T4 infos 沒有 nuScenes 的 `cam_instances` 標註)。

### 3.5 `StreamPETRHead` — 3D 主 head(整個方法的核心)
[stream_petr/models/dense_heads/streampetr_head.py](../stream_petr/models/dense_heads/streampetr_head.py)(registry:`StreamPETRHead`)

一個 DETR-style head,輸出 10-dim box 編碼 `(cx,cy,cz, log w,l,h, sin/cos yaw, vx,vy)`。先給一張**論文術語 ↔ 程式碼**對照表,下面各小節都會用到:

| 論文符號/術語 | 意義 | 程式碼 |
|---|---|---|
| memory queue(N×K = 4×256) | 歷史物體的記憶 | 5 個 buffer,各 `memory_len=1024` 條 |
| Q_c(context embedding) | 物體的特徵向量 | `memory_embedding` (B,1024,256) |
| Q_p(object center) | 物體 3D 中心 | `memory_reference_point` (B,1024,3) |
| Δt(time interval) | 距當前幀的時間差 | `memory_timestamp` |
| E(ego-pose matrix) | 記錄當時的 ego pose | `memory_egopose` (B,1024,4,4) |
| v(velocity) | 物體速度 | `memory_velo` (B,1024,2) |
| propagation transformer | 帶時序的 decoder | `PETRTemporalTransformer` |
| hybrid attention | self-attn 的 K/V 混入歷史 query | decoder layer 的 `temp_memory`/`temp_pos` |
| MLN(motion-aware LayerNorm) | 把運動資訊注入 query | [misc.py:170](../stream_petr/models/utils/misc.py#L170) 的 `MLN`,head 裡的 `ego_pose_pe`/`ego_pose_memory` |
| 3D PE(PETR 的 3D position embedding) | 給 2D 特徵 3D 身分 | `position_embeding()` + `position_encoder` |

`forward()`([L662](../stream_petr/models/dense_heads/streampetr_head.py#L662))每一幀依序做:**讀記憶 → 特徵編碼 → 準備 query(含 DN)→ 時序對齊 → decoder → 寫記憶**。以下逐段解釋。

---

**(a) Memory queue —「模型的筆記本」**

5 個 buffer 合起來就是論文的 memory queue,每行是一條「物體筆記」:*「我(特徵 Q_c)在位置 Q_p、速度 v,這是 Δt 秒前、當時車輛姿態 E 的觀測」*。1024 行 ≈ 最近 4 幀 × 每幀 256 個物體(FIFO:新的插隊頭,舊的被擠掉)。

用一個具體場景走一遍。假設 t=0 是新場景的第一幀,ego 以 10 m/s 直行,前方 10 m 有台車 A 以 8 m/s 同向行駛:

1. **t=0,`pre_update_memory()`**([L375](../stream_petr/models/dense_heads/streampetr_head.py#L375)):`prev_exists=0`,`memory_refresh` 把整本筆記乘 0 清空。但 propagated query 的機制需要「上一幀的偵測」才能運作,第一幀沒有——所以用 `pseudo_reference_points`(在偵測範圍內均勻撒的 3D 網格點,[L407-416](../stream_petr/models/dense_heads/streampetr_head.py#L407-L416))填前 256 行,當作「假筆記」讓機制照常跑。
2. **t=0 幀尾,`post_update_memory()`**([L418](../stream_petr/models/dense_heads/streampetr_head.py#L418)):decoder 輸出 900 個 query,依分類分數取 **top-256**(論文的 top-K foreground selection)。車 A 被偵測到,它的筆記(特徵、位置 (10,0,0)、速度 (8,0)、timestamp、pose)壓入隊頭。存之前用 `ego_pose` 把位置轉到**全域座標**——因為 ego 自己會動,ego 座標下的數字下一幀就失效了。
3. **t=1(0.5 秒後),`pre_update_memory()`**:ego 前進了 5 m。用 `ego_pose_inv` 把筆記裡的全域座標搬回**當前** ego 座標(論文 Eq.8-9:`E^{t-1→t} = E_t⁻¹·E_{t-1}`、`Q̃_p = E^{t-1→t}·Q_p`,code 在 [L386-390](../stream_petr/models/dense_heads/streampetr_head.py#L386-L390))。車 A 的筆記從「前方 10 m」變成「前方 5 m」。注意:這一步**假設物體靜止**,只補償了 ego 自己的運動——車 A 其實也前進了 4 m,這個誤差交給下面 (d) 的 MLN 隱式修正。

訓練時 `post_update_memory` 存的都是 `.detach()` 過的張量([L435-438](../stream_petr/models/dense_heads/streampetr_head.py#L435-L438)):**梯度不跨幀**,每個 iteration 只對當前幀反向傳播,記憶只帶「值」不帶「梯度」——這是它能用一般 batch 訓練跑時序模型的關鍵。

---

**(b) 3D Position Embedding —「給每個 pixel 一張 3D 身分證」**

問題:decoder 的 query 活在 3D 世界(reference point 是 (x,y,z)),影像特徵卻是 2D 的,cross attention 要怎麼知道「哪個 pixel 對應哪塊 3D 空間」?傳統做法是顯式 view transform(深度估計 + 投影成 BEV);PETR 的做法是反過來,**把 3D 座標編進 2D 特徵裡**。

具體數字:480×640 輸入、stride 16 → 每台相機 30×40 = 1200 個 token,5 相機共 6000 個 token。對其中一個 token(例如 CAM_FRONT 的像素 (320,240)):

1. 一個 pixel + 一個深度 = 一個 3D 點。沿這個 pixel 的視線取 `depth_num=64` 個深度值(`LID=True`:深度 bin 近密遠疏,第 1 個 bin 約 0.03 m 寬,最後一個約 1.9 m 寬——近處要準、遠處容忍誤差)。
2. 64 個 (u,v,d) 用 `lidar2img⁻¹` 反投影成 64 個 3D 點 → 攤平成 192 維向量 → `position_encoder` MLP 壓成 256 維([L451-499](../stream_petr/models/dense_heads/streampetr_head.py#L451-L499))。
3. 這個 256 維向量的語義是:*「我這個 pixel 看得到的 3D 空間,是從相機出發經過這 64 個點的一條射線」*。

之後 cross attention 裡,一個 reference point 在 (10, 5, 0) 附近的 query,自然會和「射線經過 (10,5,0)」的 pixel 相似度高——**幾何對應是用位置編碼「學」出來的,不需要顯式深度估計或 BEV 特徵圖**。這就是 PETR 系列不需要 view transform 的原因。

另外兩個 Focal-PETR 帶來的小改良:`cone`(相機內參 + 射線方向)經 `spatial_alignment`(MLN)調變影像特徵,讓特徵知道「自己是哪台相機、焦距多少」;`featurized_pe`(SE gating)讓位置編碼被特徵內容加權——白話:一個拍到天空的 pixel,它的 3D 射線編碼應該被降權,因為那條射線上沒東西。

---

**(c) Denoising(DN)query —「附答案的還原練習」**

DETR 系列的痛點:匈牙利匹配前期非常不穩定——同一台車這個 iteration 配給 query#5、下個 iteration 換 query#87,監督訊號一直跳,收斂慢。DN-DETR 的解法是加一批「不用搶匹配」的練習題。

例子:這幀 GT 有 3 台車。`prepare_for_dn()`([L545](../stream_petr/models/dense_heads/streampetr_head.py#L545))把 3 個 GT box 複製 `scalar=10` 組,每組加不同的隨機位移 → 30 個 DN query。它們的任務固定:**第 i 組的第 j 個 query 就負責還原第 j 個 GT**,不經過匈牙利匹配。位移太大的(‖noise‖ > `split=0.75`)標成背景,教模型「離 GT 太遠就該放棄」。

attention mask([L595-614](../stream_petr/models/dense_heads/streampetr_head.py#L595-L614))防止兩種作弊:正常 query 看不到 DN query(不然等於偷看答案位置);DN 各組之間互相看不到(不然第 1 組能從第 2 組推出 GT 在哪)。DN 只在訓練存在,推論時 `mask_dict=None`,一點成本都沒有。

---

**(d) Temporal alignment + hybrid attention —「把舊筆記翻譯成現在式」**

`temporal_alignment()`([L501](../stream_petr/models/dense_heads/streampetr_head.py#L501))做兩件事:

**1. 隱式運動補償(MLN)**。(a) 節說過,位置對齊只補了 ego 運動,物體自己的移動(車 A 那 4 m)還沒補。論文的做法不是顯式地「位置 += v·Δt」,而是把運動資訊餵給網路讓它自己學(消融顯示隱式比顯式好,光 ego pose 編碼就 +2.0 mAP):

```
γ = ξ₁(E^{t-1→t}, v, Δt),  β = ξ₂(E^{t-1→t}, v, Δt)     ← 兩個 linear 層
Q̃ = γ · LayerNorm(Q) + β                                  ← 條件式仿射變換
```

這就是 `MLN`(motion-aware LayerNorm)。直觀理解:普通 LayerNorm 的 scale/shift 是固定參數;MLN 的 scale/shift 由「這條 query 的運動狀態」即時算出——等於在 query 特徵上蓋一個戳記:*「我的速度 8 m/s、這筆資料是 0.5 秒前的」*,後面的 attention 層讀到戳記就能自行推算「所以它現在應該在前方 9 m 而不是 5 m」。code:`ego_pose_memory`/`ego_pose_pe` 兩個 MLN 分別調變特徵與位置編碼([L513-525](../stream_petr/models/dense_heads/streampetr_head.py#L513-L525)),運動向量先經 `nerf_positional_encoding` 頻率編碼;`time_embedding` 再把 Δt 單獨編碼加上去。

**2. Hybrid attention(論文取代 self-attention 的設計)**。1024 條 memory 分成兩種用法:

- **前 256 條(propagated queries)**:直接**串接**到 644 個新 query 後面,一起進 decoder([L530-533](../stream_petr/models/dense_heads/streampetr_head.py#L530-L533))。意義:上一幀偵測到的車 A 直接成為這一幀的「初始假設」,decoder 只需微調位置,不用從 6000 個影像 token 重新大海撈針。這也是模型能追蹤被短暫遮擋物體的原因——車 A 被卡車擋住兩幀,它的 propagated query 還在,靠 MLN 的運動外推繼續「腦補」它的位置。
- **後 768 條**:不參與預測,只作為 decoder self-attention 的**額外 key/value**(`temp_memory`/`temp_pos` 傳進每層 [petr_transformer.py](../stream_petr/models/utils/petr_transformer.py))。所以 self-attention 的 K/V = 900 個當前 query + 768 條歷史筆記 ≈ 1.7k 條——query 之間互相溝通時,順便能「查閱歷史」。這就是論文說的 hybrid attention:一個 attention 同時做 spatial(query 互斥、去重)和 temporal(參照歷史)兩件事,而成本遠低於對 6000 個影像 token 再做一次 temporal cross attention。

---

**(e) Decoder 與輸出頭**

644 新 query + 256 propagated + DN query 進 6 層 decoder(`PETRTransformerDecoder`,每層 = hybrid self-attn → cross-attn(對 6000 影像 token)→ FFN)。每層輸出都過 `cls_branches`/`reg_branches` 給出預測(6 個 branch 實際共享同一組權重,見 `_init_layers` [L277](../stream_petr/models/dense_heads/streampetr_head.py#L277))。

reg branch 是**殘差式**:query 的 reference point 在 (10, 5, 0),branch 只輸出「往哪修多少」,加回 `inverse_sigmoid(reference)` 再過 sigmoid 映回 pc_range([L713-728](../stream_petr/models/dense_heads/streampetr_head.py#L713-L728))——好處是每層 decoder 都在「上一層的答案」上迭代精修,而不是每層從零回歸。

**(f) Loss** — `loss()`([L1049](../stream_petr/models/dense_heads/streampetr_head.py#L1049)):對 6 層 decoder 各算一次(deep supervision,`d0.loss_cls`...`loss_cls`):`HungarianAssigner3D` 做一對一匹配 → focal cls loss + L1 bbox loss(`code_weights` 加權,GIoU loss 權重 0 = 不啟用)。DN query 走 `dn_loss_single`。
**T4 特有的 partial ignore**:样本的 `traffic_cone_barrier_status=False`(該片段沒標 cone/barrier)時,負樣本 query 對這兩類的分類 loss 權重設 0([_get_target_single,L825-829](../stream_petr/models/dense_heads/streampetr_head.py#L825-L829))——避免「沒標註」被當成「不存在」的錯誤監督。

**(g) 推論解碼** — `get_bboxes()`([L1166](../stream_petr/models/dense_heads/streampetr_head.py#L1166)):`NMSFreeCoder` 取最後一層輸出,全部 query×class 攤平取 top-300,不做 NMS(一對一匹配訓練下 query 天然不重複),過 `post_center_range` 與 `score_thres` 過濾;`use_bottom_center=True` 時把 z 從重心改成底面中心(Autoware 慣例)。

**(h) 論文的訓練方式 vs 本 repo 的實作**

論文用 **8 幀 sliding window** 訓練:一個 sample = 連續 8 幀,前 6 幀 backbone 走 `no_grad`(只為了把 memory 填成「有歷史」的狀態),只有後 2 幀算 loss。`Petr3D` 的 `num_frame_backbone_grads` / `num_frame_losses` 和 `obtain_history_memory()` 就是這套機制的實作。

本 repo 的 T4 訓練**沒有用 sliding window**,而是 `seq_mode=True` 的 streaming 訓練:dataset 每個 sample 只有 1 幀(T=1、`num_frame_losses=1`),「歷史」不是靠視窗內的前幾幀,而是靠 `GroupStreamingSampler` 保證同一 batch slot 連續 iteration 拿到同場景的連續幀 + head 的 memory 在 iteration 之間存活(只 detach、不清空)。等效於論文消融裡的 streaming video training——論文報告這種訓練/測試方式比 sliding window 略好(40.2 vs 39.6 mAP),而且不用一個 sample 存 8 幀影像,顯存友善。副作用是 epoch 概念被弱化:一個 epoch 內同 slot 的 memory 是連續劇,場景切換靠 `prev_exists=0` 重置。

### 3.6 Transformer 元件
[stream_petr/models/utils/petr_transformer.py](../stream_petr/models/utils/petr_transformer.py)

| 類別 | 做什麼 |
|---|---|
| `PETRTemporalTransformer`([L412](../stream_petr/models/utils/petr_transformer.py#L412)) | 只有 decoder 的 DETR transformer;負責 batch-first ↔ seq-first 轉置並把 temp_memory/temp_pos 傳進每層 |
| `PETRTemporalDecoderLayer`([L512](../stream_petr/models/utils/petr_transformer.py#L512)) | 一層 = self_attn → norm → cross_attn → norm → ffn → norm。self attention 的 key/value 會把 memory queue 串進來(hybrid attention);支援 `with_cp` checkpointing |
| `PETRMultiheadAttention`([L195](../stream_petr/models/utils/petr_transformer.py#L195)) | 標準 `nn.MultiheadAttention` 包裝(位置編碼相加後進 attention);ONNX 匯出用它 |
| `PETRMultiheadFlashAttention`([L37](../stream_petr/models/utils/petr_transformer.py#L37)) | 用 [attention.py](../stream_petr/models/utils/attention.py) 的 `FlashMHA`(flash_attn 套件,fp16)加速 cross attention;config 中 cross_attn 用 flash、self_attn 用標準版(self_attn 需要 DN attn_mask,flash attention 不支援) |

### 3.7 支援模組

- [core/bbox/assigners/hungarian_assigner_3d.py](../stream_petr/core/bbox/assigners/hungarian_assigner_3d.py):`mmdet.HungarianAssigner3D`,cost = focal cls cost + L1(normalized box, `match_costs` 加權;`match_with_velo=False` 時不比速度),scipy 匈牙利算法。2D 版多了 GIoU 和 centers2d cost。
- [core/bbox/util.py](../stream_petr/core/bbox/util.py):`normalize_bbox`/`denormalize_bbox`,定義 10-dim box 編碼(log 尺寸、sin/cos yaw)。
- [core/bbox/match_costs/match_cost.py](../stream_petr/core/bbox/match_costs/match_cost.py):4 個 cost 類(名字帶 `Assigner` 後綴避免和 mmdet 內建撞名)。
- [models/utils/misc.py](../stream_petr/models/utils/misc.py):`memory_refresh`(乘 prev_exists 清 memory)、`topk_gather`、`inverse_sigmoid`、`MLN`、`SELayer_Linear`、`locations`(2D head 的 pixel 網格)、`apply_ltrb`/`apply_center_offset`。
- [models/utils/positional_encoding.py](../stream_petr/models/utils/positional_encoding.py):`pos2posemb3d/1d`(sine 位置編碼)、`nerf_positional_encoding`(ego 運動編碼)。
- [models/utils/grid_mask.py](../stream_petr/models/utils/grid_mask.py):GridMask 影像增強(隨機格狀遮罩,prob 0.7),`Petr3D.__init__` 內硬編碼參數。
- [models/optimizer/amp.py](../stream_petr/models/optimizer/amp.py):`NoCacheAmpOptimWrapper` = AMP wrapper + `cache_enabled=False`,workaround PyTorch autocast cache 在「先 no-grad forward 再帶梯度 forward」情境下梯度遺失的 bug(StreamPETR 的多幀訓練正是這種模式)。

---

## 4. 資料層:時序 batch 是怎麼組出來的

### 4.1 `StreamPETRDataset`(T4 版)
[stream_petr/datasets/pipelines/dataset.py](../stream_petr/datasets/pipelines/dataset.py)(registry:`StreamPETRDataset`,繼承 AWML 共用的 `T4Dataset`)

- `filter_data()`:丟掉缺相機影像的 sample,依 `(scene_token, CAM_FRONT timestamp)` 排序——後續所有時序邏輯都建立在這個排序上。
- `_set_group_indices()`:`scene_token` 變化即為場景邊界,每個場景切成 `seq_split_num` 段,產生每個 sample 的 `flag`(分組 id)。`reset_origin=True` 時記下每段第一幀的 ego 位置,之後所有全域座標都相對它(避免超大座標值損失精度);timestamp 也同樣改為相對片段起點。
- `prepare_temporal_data()`:`prev_exists = (flag[i-1] == flag[i])` —— 模型靠這個 bit 決定要不要清 memory。
- `get_annot_info()`:組出 `lidar2img/intrinsics/extrinsics/ego_pose(_inv)/timestamp` 等 `collect_keys`;訓練時隨機打亂相機順序(`shuffle_cameras`);帶出 T4 特有的 `traffic_cone_barrier_status`。
- 只支援 `seq_mode=True`(單幀串流);原版 StreamPETR 的 sliding-window(`queue_length>1`)在這裡被禁用。

[nuscenes.py](../stream_petr/datasets/pipelines/nuscenes.py) 的 `StreamPETRNuScenesDataset` 是 nuScenes 版:場景邊界用「該幀沒有 lidar sweeps」判斷、2D GT 直接讀 infos 的 `cam_instances`、無 reset_origin/traffic_cone_barrier_status。

### 4.2 `GroupStreamingSampler` — 時序 batch 排程
[stream_petr/datasets/samplers/group_streaming_sampler.py](../stream_petr/datasets/samplers/group_streaming_sampler.py)(registry:`GroupStreamingSampler`)

核心設計:**batch 的每個 slot 綁定一條場景串流**。

1. 把 dataset 依 `flag` 分成若干 group(= 場景片段,組內時間有序)。
2. 每個 epoch 打亂 group 順序(組內順序永不打亂),按 rank 分配(多 GPU)。
3. 把 group 輪流塞進 `batch_size` 條 lane,然後輪流從每條 lane 的頭部彈出 index —— 所以第 k 個 batch 的 slot j 和第 k+1 個 batch 的 slot j 是**同一場景的連續兩幀**,head 中 batch 維度上每個位置的 memory 各自對應自己的場景。
4. 任何一條 lane 空了就停止(較長 lane 的尾巴被丟棄;`trim_sequences`/`pad_sequences` 控制跨 rank 對齊)。
5. `random_drop_probability`:組內隨機丟幀模擬相機掉幀(永不丟第一幀,保持 prev_exists 正確)。

例子:4 個場景 A(3 幀)、B(3 幀)、C(2 幀)、D(2 幀),`batch_size=2` →
lane0 = [A1,A2,A3,C1,C2]、lane1 = [B1,B2,B3,D1,D2],產出的 batch 序列:

| iteration | slot 0 | slot 1 | prev_exists |
|---|---|---|---|
| 1 | A1 | B1 | (0, 0) — 兩邊都是場景開頭,清 memory |
| 2 | A2 | B2 | (1, 1) — 延續,memory 傳遞 |
| 3 | A3 | B3 | (1, 1) |
| 4 | C1 | D1 | (0, 0) — 換場景,各自清 memory |
| 5 | C2 | D2 | (1, 1) |

slot 0 的 memory 從頭到尾只屬於「A 然後 C」這條串流,和 slot 1 的 B/D 互不干擾——
head 的 memory buffer 第一維就是 batch 維,每個 slot 一份。

> ⚠ **重要限制**:train set 的場景數必須 ≥ `batch_size`,否則有 lane 是空的,
> sampler 會產出 **0 個 index**,訓練迴圈空轉(每個 epoch 只存 checkpoint、不跑任何 iteration,
> log 完全沒有 train loss)。用小資料 smoke test 時把 `batch_size` 降到 ≤ 場景數。

### 4.3 Pipeline transforms
[stream_petr/datasets/pipelines/transform_3d.py](../stream_petr/datasets/pipelines/transform_3d.py)、[loading.py](../stream_petr/datasets/pipelines/loading.py)、[formating.py](../stream_petr/datasets/pipelines/formating.py)

| Transform | 做什麼 |
|---|---|
| `ResizeCropFlipRotImage` | 影像級增強(IDA):resize/crop/flip,所有 view 同參數,並把變換折進 `intrinsics`、重算 `lidar2img`——**幾何一致性由此保證** |
| `GlobalRotScaleTransImage` | BEV 空間增強:繞 z 旋轉/縮放,GT box 與 `ego_pose/extrinsics/lidar2img` 同步變換 |
| `PadMultiViewImage` / `NormalizeMultiviewImage` | pad 到 32 倍數、mmcv 標準化、HWC→CHW |
| `StreamPETRLoadAnnotations2D` | **T4 特有**:把 3D GT 投影到各相機生成 2D box/center/depth,餵 FocalHead |
| `PETRFormatBundle3D` | 全部轉 tensor;timestamp 用 float64(ego 運動計算需要精度) |

---

## 5. Deploy:3-ONNX 拆分
[deploy/torch2onnx.py](../../StreamPETR/deploy/torch2onnx.py) + [deploy/containers.py](../../StreamPETR/deploy/containers.py)

模型拆成 3 個 ONNX 分開匯出(`--section` 各跑一次;匯出前把 flash attention 換回標準 `PETRMultiheadAttention`):

1. **`extract_img_feat`**(`TrtEncoderContainer`):backbone + neck,img → img_feats。
2. **`position_embedding`**(`TrtPositionEmbeddingContainer`):3D position embedding 計算(含矩陣反投影,單獨拆出)。
3. **`pts_head_memory`**(`TrtPtsHeadContainer`):decoder + memory 更新。**狀態化**:5 個 memory buffer 變成顯式的 `pre_memory_*` 輸入 / `post_memory_*` 輸出,由 C++ 端在幀間傳遞;DN 關閉;timestamp 差值運算留在 TensorRT 外面用 float64 做。

---

## 6. Legacy / 注意事項

- `core/apis/`、`core/evaluation/`:mmcv 1.x / mmdet2 時代的訓練程式碼,**現行 mmengine 訓練完全不使用**,且依賴已不存在的模組(import 就會炸),僅供歷史參考。
- `dense_heads/` 裡的 `PETRHeadDN`、`SparseHead`、`YOLOXHeadCustom` 與 `detectors/RepDetr3D`、`backbones/EVAViT`:上游 repo 帶進來的備選元件,T4 config 未使用。
- config 的繼承鏈:`t4dataset/*.py` → `default/vov_flash_480x640_baseline.py` → AWML 共用的 `default_runtime.py` + `dataset/t4dataset/base.py`(類別定義、`name_mapping` 在這裡)。改 `info_directory_path`/`load_from` 時注意 child config 沒寫的欄位會回退到 base 的值。
- `Petr3D.train()` 有副作用(清 memory)且不回傳 self;`batch_size` 同時傳給 dataloader 和 sampler,兩者必須一致。

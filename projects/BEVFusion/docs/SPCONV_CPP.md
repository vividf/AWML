# spconv_cpp 說明：存在意義、與 cumm/spconv 的關係、對 Autoware BEVFusion 的影響

本文件說明 [autowarefoundation/spconv_cpp](https://github.com/autowarefoundation/spconv_cpp) 此一 repository 的定位、它對 **cumm** 與 **spconv** 的「改動」與影響，以及有了此 repo 後對 **Autoware BEVFusion** 訓練與部署的影響。

---

## 0. 先釐清：訓練 vs ROS2 節點各用什麼？.so 與 spconv_cpp 各是什麼？

### 0.1 AWML 訓練 BEVFusion 時，實際用到了哪些東西？

訓練時整條呼叫鏈是：

1. **AWML**：`tools/detection3d/train.py` 載入 config、建立 model（BEVFusion）、跑 PyTorch 訓練迴圈。
2. **BEVFusion 模型**：`pts_middle_encoder` = `BEVFusionSparseEncoder`，裡面用 `make_sparse_convmodule` 建出 SubMConv3d / SparseConv3d 等 **PyTorch 模組**（Python 類別）。
3. **spconv 的 Python 套件**（traveller59 的 spconv，例如 `pip install spconv-cu120`）：
   - 提供 `spconv.pytorch` 裡的 `SparseConvTensor`、`SubMConv3d`、`SparseConv3d` 等 **Python API**；
   - 這些類別的 forward 會呼叫 **C++/CUDA 實作**，而該實作就裝在 **同一個 pip 套件裡的 .so 檔** 中。
4. **pip 套件裡的 .so**：
   - 在 Linux 上通常是 `core_cc.cpython-3xx-x86_64-linux-gnu.so`（或類似名稱），放在 `site-packages/spconv/` 下；
   - 這是用 **PyTorch C++ extension / pybind11** 包起來的 **同一套 C++/CUDA 稀疏卷積程式碼**（即「libspconv」的邏輯）；
   - Python 端 `import spconv.core_cc as _ext` 後，`SubMConv3d.forward` 等就會進到這個 .so 裡執行真正的 CUDA kernel。
5. **cumm**：
   - 這套 C++/CUDA 程式碼（包含在 .so 裡）在 **編譯時** 會用到 cumm 產生的 GEMM / 稀疏卷積 kernel；
   - 用 pip 裝 spconv-cu120 時，要嘛是用 **預先編好的 wheel**（裡頭已含編譯好的 .so，cumm 的產物已打進 .so），要嘛是從原始碼建置時會依賴 cumm 產生程式碼再編譯。
   - 所以 **訓練時**：你沒有單獨「用 cumm」的步驟，而是 **spconv 的 .so 裡已經含有依賴 cumm 產生的程式碼**。

**整理：AWML 訓練 BEVFusion 時用到的與 spconv 相關部分**

| 層級 | 實際用到的東西 |
|------|----------------|
| 應用 | AWML（train.py + config + BEVFusion model） |
| 框架 | PyTorch |
| 稀疏卷積 API | **spconv 的 Python 套件**（`import spconv.pytorch`；SparseConvTensor、SubMConv3d、SparseConv3d） |
| 稀疏卷積實作 | **同一個 spconv pip 套件內建的 .so**（例如 `core_cc.*.so`），由 Python 載入並呼叫 |
| 底層 kernel | 該 .so 內含的 C++/CUDA 程式碼（編譯時依賴 cumm 產生的 kernel） |
| **是否用到 spconv_cpp？** | **否**。訓練只需 `pip install spconv-cu120`（或對應 CUDA 版本），不需要 spconv_cpp。 |

---

### 0.2 BEVFusion 在 Autoware ROS2 node 跑的時候，用到了哪些東西？

**官方 Autoware 的 BEVFusion 節點**（`autoware_universe/perception/autoware_bevfusion`，路徑如 `src/universe/autoware_universe/perception/autoware_bevfusion`）是 **C++ 實作**，推論走 **TensorRT**（或 ONNX → TensorRT），**不是** Python + PyTorch。因此：

- 節點用 **C++** 寫，用 TensorRT（或自訂 C++ pipeline）跑 BEVFusion；其中「稀疏卷積」部分不能只用 TensorRT 內建 op，需要 **traveller59 的 spconv 的 C++ 實作**（即 **libspconv**）。
- 實際需要的是 **libspconv.so**（以及必要時的 cumm 相關庫/headers），讓 C++ 節點或 TensorRT plugin 連結並呼叫。
- **traveller59 的 spconv 的 pip 套件「沒有」單獨提供獨立的 libspconv.so**：pip 裡只有「libspconv + pybind11」包成 Python extension 的 .so，是給 Python 載入用的，不能直接給純 C++ 程式連結。
- 所以要讓 **autoware_bevfusion** 跑起來，就必須取得 **可連結的 libspconv.so**，做法包括：
  - 使用 **spconv_cpp**：把預先產生好的 C++/CUDA 程式碼用 CMake 編譯，得到 **libspconv**（及 cumm .deb），安裝到系統後給 C++/ROS2 或 TensorRT 用；
  - 或使用 spconv 官方 [Pure C++ build](https://github.com/traveller59/spconv/blob/main/docs/PURE_CPP_BUILD.md) 自己產生程式碼再編譯。

也就是說：**在 Autoware 裡跑 BEVFusion 時，用的是 C++ + libspconv.so，會依賴 spconv_cpp（或等價的 Pure C++ build）建出來的庫，不會用到「Python + spconv pip 套件」那一套。**

（若有人自行以 **Python** 寫 ROS2 節點、直接載入 PyTorch 模型做推論，則會和訓練一樣需要 spconv 的 pip 套件與內建 .so；但這**不是**官方 `autoware_bevfusion` 的實作方式。）

**整理：BEVFusion 在 Autoware ROS2（autoware_bevfusion）跑時與 spconv 相關的部分**

| 項目 | 說明 |
|------|------|
| **官方實作** | `autoware_universe/perception/autoware_bevfusion`：C++ 節點 + TensorRT 推論 |
| **實際用到的** | C++、TensorRT、**libspconv.so**（及必要時 cumm） |
| **是否用到 spconv_cpp？** | **是**（要取得可連結的 libspconv，通常透過 spconv_cpp 或 Pure C++ build） |
| **是否用到 spconv 的 pip 套件？** | **否**（節點不是 Python，不載入 PyTorch 模型） |

---

### 0.3 traveller59 的 spconv 有沒有提供 .so？那是什麼、用來做什麼？

**有提供 .so，但只有「給 Python 用的那種」：**

- 你 `pip install spconv-cu120` 後，在 `site-packages/spconv/` 下會有 **Python extension 的 .so**（例如 `core_cc.cpython-310-x86_64-linux-gnu.so`）。
- 這個 .so 的內容是：**libspconv 的 C++/CUDA 實作 + pybind11 綁成 Python 模組**；也就是說，**同一份 C++ 稀疏卷積程式碼**，被編成「可被 Python import 的動態庫」。
- **用途**：讓 Python 程式（例如 AWML 訓練腳本、或 Python 寫的 ROS2 節點）在 `import spconv.pytorch` 並呼叫 `SubMConv3d(...)` 時，真正執行的是這個 .so 裡的 C++/CUDA，而不是純 Python 實作。

**traveller59 的 spconv 沒有單獨提供「獨立的 libspconv.so」：**

- pip 套件 **沒有** 一個「可讓 C++ 程式直接連結」的 `libspconv.so`；
- 若你要在 **純 C++**（例如 C++ ROS2 節點、TensorRT plugin）裡用 spconv，就必須自己從 spconv 原始碼做 **Pure C++ build**（跑 `python -m spconv.gencode` 產生程式碼再 CMake 編譯），或使用 **spconv_cpp**（見下一小節）。

---

### 0.4 spconv_cpp 會「生成」什麼、拿來做什麼用？

**spconv_cpp 本身「不再生成程式碼」：它是一份「已經生成好的」程式碼快照。**

- 上游 spconv（與 cumm）在 **建置時** 會用 Python/pccm 產生大量 C++/CUDA 檔（例如各種 GEMM kernel、sparse conv 的 .cu/.cc）。
- **spconv_cpp** 的做法是：在固定版本（cumm 0.5.3、spconv 2.3.8）與固定環境（CUDA 13.0、給定 GPU 架構列表）下，**先跑完這套產生流程**，把產出的 **所有 C++/CUDA 原始碼與 CMake 設定** 放進一個獨立 repo。
- 所以你 clone spconv_cpp 後，**裡面已經是 .cc/.cu 等檔案**，沒有 Python、沒有 pccm；你只需要用 **CMake 編譯**。

**編譯後會得到什麼（spconv_cpp 的「產物」）：**

1. **cumm**：  
   - 先編譯 `cumm/` 並打包成 **cumm_0.5.3_amd64.deb**（或 arm64 的 .deb）；  
   - 提供 headers 與庫，給後續 spconv 編譯用（或執行時連結用，視設定而定）。

2. **spconv**：  
   - 再編譯 `spconv/`，產出 **libspconv**（例如 **libspconv.so** 或靜態庫），以及對應的 headers；  
   - 可再打包成 **spconv_2.3.8_amd64.deb**（或 arm64）。

**這些產物拿來做什麼用：**

- **libspconv.so**：給 **純 C++ 程式** 或 **TensorRT** 連結使用，在 **沒有 Python** 的環境裡執行稀疏卷積（例如 C++ 寫的 Autoware ROS2 節點、車載機上的 TensorRT 推論）。
- **.deb**：方便在目標系統（x86 或 Jetson/ARM64）上 **安裝** cumm 與 spconv，不需在該機器上裝 Python 或跑程式碼產生。

**一句話**：spconv_cpp 不「生成」新演算法，而是把「本來要由 Python/pccm 生成的 C++/CUDA 程式碼」預先生成好並存成 repo；你編譯這個 repo 會得到 **libspconv.so**（和 cumm 的庫），用來在 **C++/ROS2/TensorRT** 裡跑 BEVFusion 的稀疏卷積，而不是給 Python 訓練用。

**對照總表**

| 項目 | 訓練（AWML） | Autoware ROS2（autoware_bevfusion） |
|------|--------------|-------------------------------------|
| 語言 / 執行環境 | Python + PyTorch | C++ + TensorRT（官方實作） |
| 稀疏卷積從哪來 | spconv **pip 套件**（Python API + 內建 .so） | **libspconv.so**（需透過 spconv_cpp 或 Pure C++ build 建置） |
| 用的 .so 是？ | pip 裡的 **core_cc.*.so**（Python extension） | **libspconv.so**（獨立動態庫） |
| 是否需要 spconv_cpp？ | **否** | **是** |
| cumm 的角色 | 已編進 pip 的 .so 裡（編譯時） | 編進 libspconv 或單獨 cumm .deb |

---

## 1. spconv_cpp 是什麼？存在意義為何？

### 1.1 一句話定義

**spconv_cpp** 是 **spconv**（以及其依賴的 **cumm**）的 **預先產生好的 C++/CUDA 程式碼版本**，以「**僅用 CMake 即可編譯、無需 Python 或執行期程式碼生成**」為目標，用來 **簡化散佈與部署**（ease distribution）。

### 1.2 為什麼需要這樣一個 repo？

上游 **spconv**（traveller59/spconv）與 **cumm**（FindDefinition/cumm）的建置流程大致是：

- **cumm**：以 **pccm**（Python 作為 meta-programming）在 **開發/安裝時** 產生 C++/CUDA 程式碼；可採 JIT（首次 `import cumm` 時編譯）或預先產生後再編譯。
- **spconv**：依賴 cumm，本身也含 Python 端與 C++/CUDA 端；C++/CUDA 部分同樣由 pccm/cumm 在 **建置或首次 import 時** 產生。

因此，若要在 **沒有 Python 的環境** 或 **交叉編譯（例如 ARM64/Jetson）** 下建置 spconv，就會遇到：

- 需要 Python、pccm、cumm 等完整工具鏈；
- 不同 CUDA 版本、不同 GPU 架構要重新產生/編譯，流程複雜。

**spconv_cpp** 的做法是：

- 在 **固定版本**（cumm 0.5.3、spconv 2.3.8）與 **固定環境**（CUDA 13.0、給定架構列表）下，先跑完「程式碼產生」步驟；
- 將產出的 **純 C++/CUDA 原始碼與 CMake 設定** 放進一個獨立 repo；
- 使用者只需 **clone + CMake + 編譯**，**不需執行任何 Python、不需 pccm/cumm 的產生步驟**。

所以它的存在意義可以歸納為：

1. **散佈友善**：以「預產生好的 C++/CUDA + CMake」形式散佈，降低對 Python 生態的依賴。
2. **建置環境單純**：僅需 C++ 編譯器、CUDA Toolkit、CMake，適合 Docker、CI、嵌入式或車載環境。
3. **交叉編譯**：提供 ARM64（含 Jetson Thor 等 SBSA）的 toolchain 與說明，方便在 x86 上為 ARM 建置 .deb。
4. **版本鎖定**：cumm 0.5.3、spconv 2.3.8、CUDA 13.0、架構 7.5～12.0（及 10.1/11.0 for Thor），利於 Autoware 整條鏈的再現性。

---

## 2. 它對 cumm、spconv 做了什麼「改動」與影響？

### 2.1 不是 fork、也不是對上游的 patch

spconv_cpp **沒有** 在程式邏輯上改寫 cumm 或 spconv 的演算法或 API；它也不是長期與上游 sync 的 fork。  
它其實是：

- 使用與上游 **相容的 spconv/cumm 版本**（2.3.8 / 0.5.3）；
- 執行上游提供的 **程式碼產生流程**（例如 spconv 的 pure C++ build：`python -m spconv.gencode` 等），得到一批 C++/CUDA 與 CMake 檔案；
- 將這批 **產出結果** 以獨立 repository 的形式保存並加上 **建置與打包說明**（含 amd64 與 arm64）。

因此，更精確的說法是：**spconv_cpp 是「預產生好的 spconv（與其內含的 cumm 產物）的發佈形式」**，而不是對 cumm/spconv 原始 repo 的程式碼改動。

### 2.2 對 cumm 的影響與關係

- **上游 cumm**：以 Python + pccm 產生 C++/CUDA（例如 GEMM kernel、tensor 運算等）；建置時需要 Python 或已產生的程式碼。
- **spconv_cpp 裡的 cumm**：
  - 倉庫內有獨立的 **cumm/** 目錄，對應 **cumm 0.5.3** 的 **預產生結果**（或可僅含 headers/libraries），可用 CMake 單獨編譯並打包成 `cumm_0.5.3_amd64.deb`（或 arm64）。
  - 另外，**spconv** 的原始碼樹裡已內含 **spconvlib/cumm**（例如 `spconv/src/spconvlib/cumm/gemm/main/...`），這些是 spconv 編譯時會用到的、由 cumm 產生的 GEMM 等 kernel；在 spconv_cpp 中這些同樣是「已產生好的」C++/CUDA，不再依賴執行期產生。

因此：

- **對 cumm 的「改動」**：無演算法或 API 改動；僅是 **固定版本、預先產生、並以 CMake/.deb 形式散佈**。
- **影響**：使用 spconv_cpp 的人 **不需要** 在目標環境安裝 Python/cumm/pccm，仍能得到與該版本 cumm 行為一致的程式庫（用於 spconv 的底層運算）。

### 2.3 對 spconv 的影響與關係

- **上游 spconv**：提供 Python API（如 `spconv.pytorch`）與 C++/CUDA 實作；C++/CUDA 部分可透過 `python -m spconv.gencode` 產生後再以 CMake 編譯（見 spconv 的 [PURE_CPP_BUILD.md](https://github.com/traveller59/spconv/blob/main/docs/PURE_CPP_BUILD.md)）。
- **spconv_cpp 裡的 spconv**：
  - 即為上述 **已產生好的** C++/CUDA + CMake，版本 2.3.8；
  - 建置時僅需 CMake + CUDA（以及先裝好 cumm 的 headers/libs，例如透過先建置並安裝 cumm 的 .deb）；
  - 產出為 `libspconv` 與對應 headers，可再打包成 `spconv_2.3.8_amd64.deb`（或 arm64）。

因此：

- **對 spconv 的「改動」**：同樣無演算法或 API 的邏輯改動；僅是 **固定版本、預先產生、並提供 amd64/arm64 的 CMake 與打包流程**。
- **影響**：在無 Python 的環境或嵌入式/車載工具鏈中，仍可建出與 spconv 2.3.8 相容的函式庫，供 C++/PyTorch 等整合使用。

### 2.4 版本與環境對照（來自 spconv_cpp README）

| 項目 | 內容 |
|------|------|
| cumm | 0.5.3 |
| spconv | 2.3.8 |
| CUDA | 13.0 |
| 架構列表 | 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 10.1/11.0（Thor）, 12.0 |
| 備註 | CUDA 13.0 將 Thor 的 SM101 更名為 SM110 |

---

## 3. 有了 spconv_cpp 對 Autoware BEVFusion 的影響

### 3.1 BEVFusion 與 spconv 的關係簡述

- **訓練**：AWML 的 BEVFusion 使用 **pts_middle_encoder = BEVFusionSparseEncoder**，底層依賴 **spconv**（SubMConv3d / SparseConv3d 等）；訓練時通常透過 **pip 安裝** `spconv-cuXXX`（例如 spconv-cu120），並由 Python 端呼叫 spconv。
- **部署**：BEVFusion 的 [README](../README.md) 與 [SparseConvolution 專案](../../SparseConvolution/README.md) 說明，**稀疏卷積的 ONNX 匯出與 TensorRT 推論** 需使用 **traveller59 的 spconv 後端**；部署時可能是在 **C++/ROS2**（如 [bevfusion_ros2](https://github.com/knzo25/bevfusion_ros2)）或嵌入式平台上跑推論。

因此，spconv 同時出現在 **訓練（Python/pip）** 與 **部署（C++/TensorRT/嵌入式）** 兩條鏈中。

### 3.2 訓練（Python / AWML）

- **目前常見做法**：在訓練環境安裝 `pip install spconv-cu120`（或對應 CUDA 版本），無需 spconv_cpp。
- **spconv_cpp 的影響**：  
  訓練流程 **不必** 依賴 spconv_cpp。spconv_cpp 提供的是「無 Python、僅 C++/CUDA」的建置與散佈方式，主要對 **部署與嵌入式** 有幫助；訓練仍以 pip 版 spconv 為主。

### 3.3 部署（C++、TensorRT、嵌入式、Jetson）

- **為何需要 spconv 的 C++ 庫**：  
  在車載或嵌入式環境跑 BEVFusion 推論時，若使用 TensorRT 或自訂 C++ pipeline，常需要 **libspconv**（及 cumm 相關符號）與 PyTorch/TensorRT 或 ROS2 節點連結；若該環境 **不打算** 或 **無法** 安裝 Python 與 pip 版 spconv，就必須以 **原生 C++/CUDA 庫** 形式提供 spconv。

- **spconv_cpp 帶來的影響**：  
  1. **可直接建出 libspconv（及 cumm）**：在僅有 CMake + CUDA 的環境（例如 Docker、CI、Yocto 等）中建置並安裝 .deb，無需 Python。  
  2. **支援 ARM64 交叉編譯**：透過 `extras/arm64-toolchain.cmake` 與 README 說明，可在 x86 上為 Jetson（含 Thor）等平台建置 spconv/cumm 的 .deb，方便 Autoware 在 ARM 車載機上部署 BEVFusion。  
  3. **版本與架構統一**：cumm 0.5.3、spconv 2.3.8、CUDA 13.0、多種 SM 架構一次產出，有利於 Autoware 整條軟體鏈的版本鎖定與相容性測試。

- **與 BEVFusion 部署文件的對應**：  
  - BEVFusion README 的「Sparse convolutions support」提到：部署時依 [SparseConvolution](../SparseConvolution/README.md) 專案啟用稀疏卷積，且 **僅支援 traveller59 的 spconv 後端**。  
  - 使用 spconv_cpp 建出的 **libspconv** 即為該後端的 C++ 實作；因此可視為 **滿足 Autoware BEVFusion 部署時對 spconv 的依賴** 的一種官方、可重現的建置方式。

### 3.4 小結：對 Autoware BEVFusion 的影響一覽

| 情境 | 是否依賴 spconv_cpp | 說明 |
|------|---------------------|------|
| **訓練**（Python，AWML） | 否 | 使用 pip 版 spconv（如 spconv-cu120）即可。 |
| **部署**（C++/TensorRT/ROS2，x86） | 可選但建議 | 若希望無 Python、僅以 .deb 提供 libspconv，可用 spconv_cpp 建置並安裝。 |
| **部署**（Jetson/ARM64） | 建議 | 需 ARM64 的 spconv/cumm；spconv_cpp 提供交叉編譯與 .deb 流程，與 Autoware 部署需求對齊。 |
| **版本與再現性** | 有幫助 | 固定 cumm 0.5.3、spconv 2.3.8、CUDA 13.0，利於 Autoware 堆疊的測試與釋出。 |

---

## 4. 總結

- **spconv_cpp 的意義**：它是 **spconv（及 cumm）的預產生 C++/CUDA 發佈版**，目標是 **不需 Python、僅用 CMake 即可編譯與散佈**，並支援 amd64 與 arm64（含 Jetson Thor）。
- **對 cumm / spconv 的「改動」**：沒有改寫上游邏輯；而是 **固定版本、預先跑完程式碼產生、並以 CMake/.deb 形式提供**，讓建置與部署不再依賴 Python/pccm/cumm 的執行期產生。
- **對 Autoware BEVFusion**：  
  - **訓練**：無需 spconv_cpp，繼續使用 pip 版 spconv 即可。  
  - **部署**：若要在 C++/嵌入式/Jetson 上使用 traveller59 的 spconv 後端，spconv_cpp 提供可重現的建置與 .deb 安裝方式，並支援 ARM64 交叉編譯，對 Autoware BEVFusion 的實際執行（尤其是車載與 TensorRT 推論）有直接幫助。

更多建置步驟與選項請見 [spconv_cpp 官方 README](https://github.com/autowarefoundation/spconv_cpp)；BEVFusion 訓練與 sparse encoder 細節請見本目錄的 [TRAINING.md](TRAINING.md)。

#!/usr/bin/env python
"""
顯示 MSE calibration 方法的完整調用鏈。

這個腳本會追蹤從 centerpoint_quantization.py 到實際 MSE 實現的完整路徑。
"""

import inspect
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def show_call_chain():
    """顯示 MSE calibration 的完整調用鏈"""
    print("=" * 80)
    print("MSE Calibration 調用鏈")
    print("=" * 80)

    print("\n1. 入口點：tools/detection3d/centerpoint_quantization.py")
    print("   " + "-" * 76)
    print("   calibrator.calibrate(")
    print("       dataloader,")
    print("       num_batches=args.calibrate_batches,")
    print("       method='mse',  # ← 這裡指定 MSE 方法")
    print("   )")

    print("\n2. CalibrationManager.calibrate()")
    print("   位置：projects/CenterPoint/quantization/calibration/calibrator.py")
    print("   " + "-" * 76)
    print("   def calibrate(self, dataloader, num_batches=100, method='mse'):")
    print("       self.set_quantizer_fast()")
    print("       self.collect_stats(dataloader, num_batches, forward_fn)")
    print("       self.compute_amax(method)  # ← 調用 compute_amax")

    print("\n3. CalibrationManager.compute_amax()")
    print("   位置：projects/CenterPoint/quantization/calibration/calibrator.py:145")
    print("   " + "-" * 76)
    print("   def compute_amax(self, method='mse'):")
    print("       for name, module in self.model.named_modules():")
    print("           if isinstance(module, TensorQuantizer):")
    print("               if module._calibrator is not None:")
    print("                   module.load_calib_amax(method=method)  # ← 調用 TensorQuantizer")

    print("\n4. TensorQuantizer.load_calib_amax()")
    print("   位置：pytorch-quantization 庫")
    print("   導入：from pytorch_quantization.nn import TensorQuantizer")
    print("   " + "-" * 76)
    print("   這是 pytorch-quantization 庫中的方法")
    print("   實際實現位置：pytorch_quantization/nn/modules/tensor_quantizer.py")

    # 嘗試導入並查看實際的方法簽名
    try:
        from pytorch_quantization import calib
        from pytorch_quantization.nn import TensorQuantizer

        print("\n5. 實際的 MSE 實現")
        print("   " + "-" * 76)

        # 查看 TensorQuantizer 的 load_calib_amax 方法
        if hasattr(TensorQuantizer, "load_calib_amax"):
            sig = inspect.signature(TensorQuantizer.load_calib_amax)
            print(f"   TensorQuantizer.load_calib_amax{sig}")
            print(f"   文檔：{TensorQuantizer.load_calib_amax.__doc__}")

        # 查看 HistogramCalibrator（這是實際執行 MSE 的類）
        print("\n   實際執行 MSE 的類：HistogramCalibrator")
        print("   位置：pytorch_quantization/calib/histogram.py")

        if hasattr(calib, "HistogramCalibrator"):
            print(f"   類：{calib.HistogramCalibrator}")
            # 嘗試查看相關方法
            if hasattr(calib.HistogramCalibrator, "load_calib_amax"):
                print(f"   方法：load_calib_amax")

        print("\n   MSE 計算邏輯（簡化版）：")
        print("   " + "-" * 76)
        print("   1. 從 histogram 中讀取統計數據")
        print("   2. 對每個可能的 amax 候選值：")
        print("      - 計算量化 scale = amax / (2^(num_bits-1) - 1)")
        print("      - 對每個 histogram bin，計算量化誤差")
        print("      - 誤差 = (原始值 - 量化值)^2")
        print("      - 總誤差 = sum(histogram[i] * error[i])")
        print("   3. 選擇總誤差最小的 amax 值")

    except ImportError as e:
        print(f"\n   無法導入 pytorch-quantization: {e}")
        print("   請確保已安裝：pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")

    print("\n" + "=" * 80)
    print("完整調用鏈總結")
    print("=" * 80)
    print(
        """
1. centerpoint_quantization.py
   └─> calibrator.calibrate(method="mse")

2. calibrator.py (CalibrationManager)
   └─> self.compute_amax(method="mse")

3. calibrator.py (CalibrationManager.compute_amax)
   └─> module.load_calib_amax(method="mse")
       (module 是 TensorQuantizer 實例)

4. pytorch-quantization 庫
   └─> TensorQuantizer.load_calib_amax(method="mse")
       └─> HistogramCalibrator.load_calib_amax(method="mse")
           └─> 實際的 MSE 優化算法
               - 遍歷 amax 候選值
               - 計算量化誤差
               - 選擇誤差最小的 amax
    """
    )

    print("\n實際代碼位置：")
    print("  - AWML 代碼：projects/CenterPoint/quantization/calibration/calibrator.py")
    print("  - 庫代碼：pytorch_quantization/nn/modules/tensor_quantizer.py")
    print("  - 庫代碼：pytorch_quantization/calib/histogram.py")
    print("=" * 80)


def show_source_locations():
    """顯示相關源代碼的位置"""
    print("\n" + "=" * 80)
    print("相關源代碼位置")
    print("=" * 80)

    locations = {
        "AWML 代碼": [
            "tools/detection3d/centerpoint_quantization.py:337",
            "projects/CenterPoint/quantization/calibration/calibrator.py:145-166",
        ],
        "pytorch-quantization 庫": [
            "pytorch_quantization/nn/modules/tensor_quantizer.py:load_calib_amax()",
            "pytorch_quantization/calib/histogram.py:HistogramCalibrator.load_calib_amax()",
        ],
    }

    for category, files in locations.items():
        print(f"\n{category}:")
        for file in files:
            print(f"  - {file}")

    print("\n查看 pytorch-quantization 源代碼的方法：")
    print("  1. 找到 Python 環境中的庫位置：")
    print("     python -c 'import pytorch_quantization; print(pytorch_quantization.__file__)'")
    print("  2. 查看源代碼：")
    print("     cat <path>/pytorch_quantization/calib/histogram.py")
    print("=" * 80)


if __name__ == "__main__":
    show_call_chain()
    show_source_locations()

    # 嘗試找到實際的庫位置
    print("\n" + "=" * 80)
    print("查找 pytorch-quantization 庫位置")
    print("=" * 80)
    try:
        import os

        import pytorch_quantization

        lib_path = os.path.dirname(pytorch_quantization.__file__)
        print(f"\npytorch-quantization 庫位置：{lib_path}")
        print(f"\n相關文件：")
        print(f"  - {lib_path}/nn/modules/tensor_quantizer.py")
        print(f"  - {lib_path}/calib/histogram.py")
        print(f"\n查看 MSE 實現：")
        print(f"  cat {lib_path}/calib/histogram.py | grep -A 50 'def load_calib_amax'")
    except ImportError:
        print("\n無法找到 pytorch-quantization 庫")

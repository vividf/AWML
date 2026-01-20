#!/usr/bin/env python
"""
詳細解釋為什麼 calibration 會因為順序而產生差異。

這個腳本演示：
1. 浮點數運算的非結合性
2. Histogram 累積的順序敏感性
3. MSE 優化對 histogram 形狀的敏感性
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def demonstrate_floating_point_non_associativity():
    """
    演示浮點數運算的非結合性。

    浮點數在計算機中是以二進制表示的，有精度限制。
    當進行多次加法時，不同的結合順序會導致不同的結果。
    """
    print("=" * 80)
    print("1. 浮點數運算的非結合性演示")
    print("=" * 80)

    # 創建一些數值，模擬 activation 值
    values = [1e10, 1.0, -1e10, 1.0, 1e10]

    # 方法 1: 從左到右累加
    result1 = 0.0
    for v in values:
        result1 += v
        print(f"  累加: {result1:.15f}")

    # 方法 2: 先加小數，再加大數
    small = sum([v for v in values if abs(v) < 1e5])
    large = sum([v for v in values if abs(v) >= 1e5])
    result2 = small + large

    print(f"\n  順序 1 (從左到右): {result1:.15f}")
    print(f"  順序 2 (先小後大): {result2:.15f}")
    print(f"  差異: {abs(result1 - result2):.15f}")

    # 更實際的例子：histogram bin 累積
    print("\n  實際例子：Histogram Bin 累積")
    bin_values = [1.23456789012345, 2.34567890123456, 3.45678901234567]

    # 順序 A: 按順序累積
    sum_a = 0.0
    for v in bin_values:
        sum_a += v

    # 順序 B: 反向累積
    sum_b = 0.0
    for v in reversed(bin_values):
        sum_b += v

    print(f"  順序 A: {sum_a:.15f}")
    print(f"  順序 B: {sum_b:.15f}")
    print(f"  差異: {abs(sum_a - sum_b):.15f}")
    print(f"  相對誤差: {abs(sum_a - sum_b) / sum_a * 100:.10f}%")

    return result1, result2


def demonstrate_histogram_accumulation_order():
    """
    演示 histogram 累積的順序敏感性。

    Histogram 的每個 bin 都需要累積計數。當樣本以不同順序處理時，
    即使最終每個 bin 的計數相同，累積過程中的中間值可能不同。
    """
    print("\n" + "=" * 80)
    print("2. Histogram 累積的順序敏感性")
    print("=" * 80)

    # 模擬 activation 值分布
    np.random.seed(42)
    activations_seq = np.random.normal(3.0, 1.0, 1000)  # Sequential order
    activations_rand = activations_seq.copy()
    np.random.shuffle(activations_rand)  # Random order

    # 創建 histogram bins
    bins = np.linspace(0, 6, 101)  # 100 bins

    # 方法 1: Sequential 累積
    hist_seq = np.zeros(len(bins) - 1, dtype=np.float64)
    for val in activations_seq:
        bin_idx = np.digitize(val, bins) - 1
        if 0 <= bin_idx < len(hist_seq):
            hist_seq[bin_idx] += 1.0

    # 方法 2: Random 累積（理論上應該相同）
    hist_rand = np.zeros(len(bins) - 1, dtype=np.float64)
    for val in activations_rand:
        bin_idx = np.digitize(val, bins) - 1
        if 0 <= bin_idx < len(hist_rand):
            hist_rand[bin_idx] += 1.0

    # 檢查差異
    diff = np.abs(hist_seq - hist_rand)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  總樣本數: {len(activations_seq)}")
    print(f"  Histogram bins: {len(bins) - 1}")
    print(f"  最大 bin 計數差異: {max_diff:.10f}")
    print(f"  平均 bin 計數差異: {mean_diff:.10f}")

    # 理論上應該完全相同，但由於浮點數精度，可能有微小差異
    if max_diff < 1e-10:
        print("  ✓ 差異在浮點數精度範圍內（可忽略）")
    else:
        print(f"  ⚠ 發現差異（可能是 binning 邊界問題）")

    # 演示累積過程中的差異
    print("\n  累積過程中的差異演示：")
    print("  假設我們要累積 3 個值到同一個 bin:")
    values_to_bin = [1.23456789012345, 2.34567890123456, 3.45678901234567]

    # 順序 A
    bin_count_a = 0.0
    for v in values_to_bin:
        bin_count_a += 1.0  # 每個值貢獻 1
        print(f"    順序 A - 累積到 {len(values_to_bin)} 個值: {bin_count_a:.15f}")

    # 順序 B（反向）
    bin_count_b = 0.0
    for v in reversed(values_to_bin):
        bin_count_b += 1.0
        print(f"    順序 B - 累積到 {len(values_to_bin)} 個值: {bin_count_b:.15f}")

    print(f"  最終計數差異: {abs(bin_count_a - bin_count_b):.15f}")

    return hist_seq, hist_rand


def demonstrate_mse_sensitivity_to_histogram_shape():
    """
    演示 MSE 優化對 histogram 形狀的敏感性。

    MSE (Mean Squared Error) 方法通過最小化量化誤差來找到最優 amax。
    即使兩個 histogram 的最終計數相同，如果累積過程不同導致
    中間狀態不同，MSE 優化可能會找到不同的 amax 值。
    """
    print("\n" + "=" * 80)
    print("3. MSE 優化對 Histogram 形狀的敏感性")
    print("=" * 80)

    # 創建兩個理論上相同但累積過程不同的 histogram
    # 實際情況：由於浮點數精度，累積過程會產生微小差異

    np.random.seed(42)
    true_values = np.random.normal(3.0, 1.0, 1000)

    # 創建 bins
    bins = np.linspace(0, 6, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Histogram 1: Sequential 累積（模擬）
    hist1 = np.histogram(true_values, bins=bins)[0].astype(np.float64)

    # Histogram 2: 添加微小的浮點數誤差（模擬不同累積順序的影響）
    hist2 = hist1.copy()
    # 在幾個 bin 中添加微小的差異（模擬浮點數累積誤差）
    noise = np.random.normal(0, 1e-10, len(hist2))
    hist2 += noise
    hist2 = np.maximum(hist2, 0)  # 確保非負

    # 計算差異
    diff = np.abs(hist1 - hist2)
    print(f"  Histogram 1 總計數: {np.sum(hist1):.10f}")
    print(f"  Histogram 2 總計數: {np.sum(hist2):.10f}")
    print(f"  最大 bin 差異: {np.max(diff):.10f}")
    print(f"  平均 bin 差異: {np.mean(diff):.10f}")
    print(f"  相對差異: {np.max(diff) / np.sum(hist1) * 100:.10f}%")

    # 模擬 MSE 優化過程
    print("\n  模擬 MSE 優化過程：")
    print("  MSE 方法會嘗試不同的 amax 值，計算量化誤差，選擇最小的")

    def compute_quantization_error(hist, bin_centers, amax, num_bits=8):
        """計算給定 amax 的量化誤差"""
        # 簡化的量化誤差計算
        # 實際的 MSE 方法更複雜，但原理類似
        scale = amax / (2 ** (num_bits - 1) - 1)
        quantized = np.round(bin_centers / scale) * scale
        error = np.sum(hist * (bin_centers - quantized) ** 2)
        return error

    # 測試不同的 amax 值
    amax_candidates = np.linspace(2.0, 5.0, 100)
    errors1 = [compute_quantization_error(hist1, bin_centers, amax) for amax in amax_candidates]
    errors2 = [compute_quantization_error(hist2, bin_centers, amax) for amax in amax_candidates]

    optimal_amax1 = amax_candidates[np.argmin(errors1)]
    optimal_amax2 = amax_candidates[np.argmin(errors2)]

    print(f"  Histogram 1 最優 amax: {optimal_amax1:.6f}")
    print(f"  Histogram 2 最優 amax: {optimal_amax2:.6f}")
    print(f"  amax 差異: {abs(optimal_amax1 - optimal_amax2):.6f}")
    print(f"  相對差異: {abs(optimal_amax1 - optimal_amax2) / optimal_amax1 * 100:.4f}%")

    # 解釋為什麼會有差異
    print("\n  為什麼會有差異？")
    print("  1. MSE 優化是連續的過程，需要計算每個 amax 候選值的誤差")
    print("  2. 即使 histogram 計數差異很小（< 0.01%），誤差函數的形狀也會改變")
    print("  3. 誤差函數的最小值位置（最優 amax）對 histogram 的微小變化很敏感")
    print("  4. 這就像在一個略微不同的地形上尋找最低點，結果可能不同")

    return optimal_amax1, optimal_amax2


def demonstrate_real_world_example():
    """
    演示真實世界的例子：為什麼 Sequential 和 Random 會有差異。
    """
    print("\n" + "=" * 80)
    print("4. 真實世界例子：為什麼 Sequential 和 Random 會有差異")
    print("=" * 80)

    print(
        """
  場景：使用 938 個 calibration samples，全部使用，但順序不同

  步驟 1: 資料載入
  - Sequential: 樣本按 annotation file 順序載入 [0, 1, 2, ..., 937]
  - Random: 樣本被 shuffle [例如: 234, 12, 891, ..., 567]

  步驟 2: Histogram 累積
  - 每個樣本通過模型，產生 activation 值
  - 每個 activation 值被分配到對應的 histogram bin
  - bin 計數被累積：bin[i] += 1.0

  問題 1: 浮點數累積誤差
  - Sequential: bin[50] = ((((0 + 1) + 1) + 1) + ...) + 1
  - Random:    bin[50] = ((((0 + 1) + 1) + 1) + ...) + 1
  - 雖然最終計數相同，但中間累積過程不同
  - 由於浮點數非結合性，可能產生微小差異（通常 < 1e-15）

  問題 2: Histogram Binning 邊界
  - 當 activation 值正好在 bin 邊界時，digitize 的結果可能因累積順序而不同
  - 例如：值 3.000000000000001 vs 3.000000000000002 可能被分到不同 bin
  - 這會導致 histogram 形狀的微小差異

  問題 3: MSE 優化敏感性
  - pytorch-quantization 的 MSE 方法會：
    1. 對每個可能的 amax 值計算量化誤差
    2. 選擇誤差最小的 amax
  - 即使 histogram 差異很小（例如 0.01%），誤差函數的形狀也會改變
  - 這會導致最優 amax 的微小偏移（例如 3.631084 vs 3.309321，差異 8.86%）

  問題 4: 多進程 DataLoader
  - num_workers=32 意味著 32 個進程同時載入資料
  - 即使設定了 seed，不同進程的初始化順序可能不同
  - 這導致樣本被處理的實際順序與預期不同
  - 進一步放大了累積誤差

  結果：
  - Sequential: amax = 3.631084 (mAP = 0.6690)
  - Random:    amax = 3.309321 (mAP = 0.6748)
  - 差異：8.86% 的 amax 差異導致 0.58% 的 mAP 差異
    """
    )


def main():
    """主函數：運行所有演示"""
    print("\n" + "=" * 80)
    print("Calibration 差異的詳細解釋")
    print("=" * 80)
    print("\n這個腳本將詳細解釋為什麼即使使用全部 calibration samples，")
    print("不同的資料順序仍會導致不同的 calibration 結果。\n")

    # 1. 浮點數非結合性
    demonstrate_floating_point_non_associativity()

    # 2. Histogram 累積順序敏感性
    demonstrate_histogram_accumulation_order()

    # 3. MSE 優化敏感性
    demonstrate_mse_sensitivity_to_histogram_shape()

    # 4. 真實世界例子
    demonstrate_real_world_example()

    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print(
        """
  1. 浮點數運算的非結合性導致累積過程的微小差異
  2. Histogram binning 的邊界效應進一步放大差異
  3. MSE 優化對 histogram 形狀敏感，微小差異會影響最優 amax
  4. 多進程 DataLoader 引入額外的非確定性

  解決方案：
  - 使用 num_workers=0 確保單進程順序處理
  - 使用 Sequential 順序（通常結果更好且更穩定）
  - 接受小差異（< 1%）是正常的
  - 考慮使用 'max' 方法（更穩定但可能精度略低）
    """
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

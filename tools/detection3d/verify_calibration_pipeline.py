#!/usr/bin/env python
"""
驗證 calibration pipeline 是否包含任何 random augmentation。

這個腳本會直接讀取配置文件並檢查。
"""

import re
from pathlib import Path


def check_config_file(config_path: str):
    """直接讀取配置文件並檢查"""
    print("=" * 80)
    print("Calibration Pipeline 檢查")
    print("=" * 80)

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return

    content = config_path.read_text()

    print(f"\n配置文件: {config_path}")

    # 檢查 val_dataloader 使用的 pipeline
    print("\n1. 檢查 val_dataloader 配置:")
    print("-" * 80)

    # 找到 val_dataloader 區塊
    val_dataloader_match = re.search(r"val_dataloader\s*=\s*dict\(([^)]+)\)", content, re.DOTALL)
    if val_dataloader_match:
        val_dataloader_content = val_dataloader_match.group(1)

        # 檢查使用的 pipeline
        pipeline_match = re.search(r"pipeline\s*=\s*(\w+)", val_dataloader_content)
        if pipeline_match:
            pipeline_name = pipeline_match.group(1)
            print(f"   ✅ val_dataloader 使用: {pipeline_name}")
        else:
            print("   ⚠️  無法找到 pipeline 配置")

        # 檢查 test_mode
        test_mode_match = re.search(r"test_mode\s*=\s*(\w+)", val_dataloader_content)
        if test_mode_match:
            test_mode = test_mode_match.group(1)
            print(f"   ✅ test_mode: {test_mode}")
        else:
            print("   ⚠️  test_mode 未設置")

    # 檢查 test_pipeline
    print("\n2. 檢查 test_pipeline:")
    print("-" * 80)

    test_pipeline_match = re.search(r"test_pipeline\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if test_pipeline_match:
        test_pipeline_content = test_pipeline_match.group(1)

        # 檢查是否有 Random 相關的 transform
        random_patterns = [
            r"RandomFlip",
            r"RandomRot",
            r"RandomScale",
            r"Random.*",
            r"PointShuffle",
            r"GlobalRotScaleTrans.*rot_range",  # 如果有 random rotation
        ]

        found_random = []
        for pattern in random_patterns:
            matches = re.findall(pattern, test_pipeline_content, re.IGNORECASE)
            if matches:
                found_random.extend(matches)

        if found_random:
            print(f"   ❌ 發現 Random Transform: {found_random}")
        else:
            print("   ✅ 沒有發現 Random Transform")

        # 檢查 LoadPointsFromMultiSweeps 的 test_mode
        multisweep_match = re.search(
            r"LoadPointsFromMultiSweeps.*?test_mode\s*=\s*(\w+)", test_pipeline_content, re.DOTALL
        )
        if multisweep_match:
            multisweep_test_mode = multisweep_match.group(1)
            if multisweep_test_mode.lower() == "true":
                print(f"   ✅ LoadPointsFromMultiSweeps.test_mode: {multisweep_test_mode} (確定性)")
            else:
                print(f"   ❌ LoadPointsFromMultiSweeps.test_mode: {multisweep_test_mode} (非確定性！)")
        else:
            print("   ⚠️  無法確認 LoadPointsFromMultiSweeps.test_mode")

    # 對比 train_pipeline（看看有什麼不同）
    print("\n3. 對比 train_pipeline（參考）:")
    print("-" * 80)

    train_pipeline_match = re.search(r"train_pipeline\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if train_pipeline_match:
        train_pipeline_content = train_pipeline_match.group(1)

        train_random = []
        for pattern in random_patterns:
            matches = re.findall(pattern, train_pipeline_content, re.IGNORECASE)
            if matches:
                train_random.extend(matches)

        if train_random:
            print(f"   train_pipeline 包含: {set(train_random)}")
            print("   （這是正常的，train 時需要 augmentation）")
        else:
            print("   train_pipeline 沒有 Random Transform（不常見）")

    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)

    # 最終判斷
    if found_random:
        print(
            """
❌ 發現問題！

Calibration 時使用的 pipeline 包含 Random Transform。
這會導致：
  1. 每次運行時，即使使用相同的樣本，輸入到模型的 tensor 也不同
  2. Calibration 看到的 activation 分布不一致
  3. 導致 quantizer scale 差異很大（可能 > 5%）

解決方案：
  1. 確保 val_dataloader 使用 test_pipeline（不是 train_pipeline）
  2. 確保 test_pipeline 中沒有 Random* transform
  3. 確保 LoadPointsFromMultiSweeps 有 test_mode=True
        """
        )
    else:
        print(
            """
✅ Pipeline 看起來正確！

val_dataloader 使用 test_pipeline，且沒有 Random Transform。
如果仍然有 calibration 差異，可能是：
  1. 浮點數累積誤差（順序相關）
  2. MSE 優化對 histogram 形狀敏感
  3. 多進程 DataLoader 的非確定性
        """
        )

    print("=" * 80)


if __name__ == "__main__":
    import sys

    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py"
    )

    check_config_file(config_path)

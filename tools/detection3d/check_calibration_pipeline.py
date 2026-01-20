#!/usr/bin/env python
"""
檢查 calibration 時使用的 pipeline 是否包含 random augmentation。

這個腳本會：
1. 檢查 val_dataloader 使用的 pipeline
2. 列出所有包含 "Random" 或 "random" 的 transform
3. 確認是否應該使用純 eval/test pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mmengine.config import Config


def check_pipeline_for_random_transforms(config_path: str):
    """檢查 pipeline 中是否包含 random transforms"""
    print("=" * 80)
    print("檢查 Calibration Pipeline")
    print("=" * 80)

    cfg = Config.fromfile(config_path)

    # 檢查 val_dataloader 使用的 pipeline
    val_dataloader = cfg.val_dataloader
    pipeline = val_dataloader.dataset.pipeline

    print(f"\nConfig 文件: {config_path}")
    print(f"val_dataloader 使用的 pipeline: {type(pipeline).__name__}")

    if isinstance(pipeline, list):
        print(f"Pipeline 長度: {len(pipeline)}")
        print("\nPipeline 內容:")
        print("-" * 80)

        random_transforms = []
        for i, transform in enumerate(pipeline):
            transform_type = transform.get("type", "Unknown")
            print(f"  {i+1}. {transform_type}")

            # 檢查是否包含 random
            if "random" in transform_type.lower() or "Random" in transform_type:
                random_transforms.append((i, transform_type, transform))
                print(f"     ⚠️  發現 Random Transform!")

        print("-" * 80)

        if random_transforms:
            print(f"\n❌ 發現 {len(random_transforms)} 個 Random Transform:")
            for idx, transform_type, transform in random_transforms:
                print(f"  - [{idx}] {transform_type}")
                print(f"    參數: {transform}")
            print("\n⚠️  警告：Calibration 時不應該使用 Random Transform！")
            print("   這會導致每次運行時輸入到模型的 tensor 不同，")
            print("   進而導致 calibration 結果不一致。")
        else:
            print("\n✅ 沒有發現 Random Transform")
            print("   Pipeline 看起來是純 eval/test pipeline")

        # 檢查其他可能的隨機性來源
        print("\n檢查其他可能的隨機性來源:")
        print("-" * 80)

        # 檢查 PointShuffle
        point_shuffle = [t for t in pipeline if "PointShuffle" in t.get("type", "")]
        if point_shuffle:
            print("  ⚠️  發現 PointShuffle:")
            for t in point_shuffle:
                print(f"     - {t}")
        else:
            print("  ✅ 沒有 PointShuffle")

        # 檢查 sampler shuffle
        if "sampler" in val_dataloader:
            sampler = val_dataloader.sampler
            if isinstance(sampler, dict):
                shuffle = sampler.get("shuffle", False)
                if shuffle:
                    print(f"  ⚠️  Sampler shuffle: {shuffle}")
                else:
                    print(f"  ✅ Sampler shuffle: {shuffle}")
        elif "shuffle" in val_dataloader:
            shuffle = val_dataloader.shuffle
            if shuffle:
                print(f"  ⚠️  Dataloader shuffle: {shuffle}")
            else:
                print(f"  ✅ Dataloader shuffle: {shuffle}")

        # 檢查 test_mode
        test_mode = val_dataloader.dataset.get("test_mode", None)
        if test_mode:
            print(f"  ✅ test_mode: {test_mode}")
        else:
            print(f"  ⚠️  test_mode 未設置")

    print("\n" + "=" * 80)
    print("建議")
    print("=" * 80)
    print(
        """
Calibration 時應該使用純 eval/test pipeline，不應該包含：
  - RandomFlip3D
  - GlobalRotScaleTrans (如果包含 random rotation/scale)
  - PointShuffle
  - 任何其他 random augmentation

如果發現 random transforms，應該：
  1. 確保 val_dataloader 使用 test_pipeline 而不是 train_pipeline
  2. 檢查 test_pipeline 是否真的沒有 random transforms
  3. 如果必須使用 train_pipeline，需要創建一個 calibration_pipeline
     （移除所有 random transforms）
    """
    )
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="檢查 calibration pipeline")
    parser.add_argument(
        "config",
        nargs="?",
        default="projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_t4metric_v2.py",
        help="Config 文件路徑",
    )

    args = parser.parse_args()
    check_pipeline_for_random_transforms(args.config)

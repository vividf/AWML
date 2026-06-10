#!/usr/bin/env python3

import argparse
import math
import sqlite3
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rclpy
from autoware_internal_debug_msgs.msg import Float64Stamped
from rclpy.serialization import deserialize_message

DEFAULT_ROSBAG_ROOT = Path("/media/yihsiangfang/VIVID/model/bevfusion_2_7/rosbag/rosbag_release_comparison")

METRIC_TOPICS = {
    "inference_ms": "/perception/object_recognition/detection/bevfusion/bevfusion/debug/processing_time/inference_ms",
    "preprocess_ms": "/perception/object_recognition/detection/bevfusion/bevfusion/debug/processing_time/preprocess_ms",
    "postprocess_ms": "/perception/object_recognition/detection/bevfusion/bevfusion/debug/processing_time/postprocess_ms",
    "total_ms": "/perception/object_recognition/detection/bevfusion/bevfusion/debug/processing_time/total_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BEVFusion latency topics across multiple ROS2 bags.")
    parser.add_argument(
        "--rosbag-root",
        type=Path,
        default=DEFAULT_ROSBAG_ROOT,
        help="Root directory containing bag folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./bevfusion_latency_analysis_slow"),
        help="Directory for generated CSV and plots.",
    )
    parser.add_argument(
        "--bags",
        nargs="+",
        default=None,
        help="Optional bag names to include. Example: --bags 2_7_bag_slow 2_7_opt_bag_slow",
    )
    return parser.parse_args()


def resolve_db3_path(bag_dir: Path) -> Path:
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag directory does not exist: {bag_dir}")

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_dir}")
    if len(db3_files) > 1:
        raise RuntimeError(f"Expected one .db3 file in {bag_dir}, found: {db3_files}")
    return db3_files[0]


def discover_bag_dirs(rosbag_root: Path) -> Dict[str, Path]:
    if not rosbag_root.exists():
        raise FileNotFoundError(f"Rosbag root does not exist: {rosbag_root}")

    bag_dirs = sorted([p for p in rosbag_root.iterdir() if p.is_dir()])
    if not bag_dirs:
        raise RuntimeError(f"No bag directories found in {rosbag_root}")

    discovered = {}
    for bag_dir in bag_dirs:
        if any(bag_dir.glob("*.db3")):
            discovered[bag_dir.name] = bag_dir

    if not discovered:
        raise RuntimeError(f"No bag directory with .db3 found in {rosbag_root}")

    return discovered


def resolve_requested_bags(discovered: Dict[str, Path], requested_bags: List[str]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    available_names = list(discovered.keys())

    for requested in requested_bags:
        if requested in discovered:
            resolved[requested] = discovered[requested]
            continue

        substring_matches = [name for name in available_names if requested in name or name in requested]
        if len(substring_matches) == 1:
            chosen = substring_matches[0]
            resolved[chosen] = discovered[chosen]
            continue
        if len(substring_matches) > 1:
            raise RuntimeError(f"Ambiguous bag name '{requested}'. Candidates: {', '.join(substring_matches)}")

        close_matches = get_close_matches(requested, available_names, n=1, cutoff=0.6)
        if close_matches:
            chosen = close_matches[0]
            print(f"[Info] Use closest bag name '{chosen}' for requested '{requested}'")
            resolved[chosen] = discovered[chosen]
            continue

        raise RuntimeError(f"Requested bag '{requested}' not found. Available: {', '.join(sorted(available_names))}")

    return resolved


def read_float64_stamped_from_rosbag(db3_path: Path, topic_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM topics WHERE name = ?", (topic_name,))
    row = cur.fetchone()

    if row is None:
        available_topics = cur.execute("SELECT name FROM topics").fetchall()
        conn.close()
        available_topics = [t[0] for t in available_topics]
        raise RuntimeError(
            f"Topic not found: {topic_name}\n" f"Bag: {db3_path}\n" "Available topics:\n" + "\n".join(available_topics)
        )

    topic_id, _ = row

    cur.execute(
        """
        SELECT timestamp, data
        FROM messages
        WHERE topic_id = ?
        ORDER BY timestamp
        """,
        (topic_id,),
    )
    rows = cur.fetchall()
    conn.close()

    timestamps_ns = []
    values_ms = []
    for timestamp_ns, serialized_data in rows:
        msg = deserialize_message(serialized_data, Float64Stamped)
        timestamps_ns.append(timestamp_ns)
        values_ms.append(float(msg.data))

    if not values_ms:
        raise RuntimeError(f"No messages found for topic {topic_name} in {db3_path}")

    timestamps_ns_arr = np.array(timestamps_ns, dtype=np.int64)
    values_ms_arr = np.array(values_ms, dtype=np.float64)
    time_from_start_s = (timestamps_ns_arr - timestamps_ns_arr[0]) * 1e-9

    return pd.DataFrame(
        {
            "sample_index": np.arange(len(values_ms_arr), dtype=np.int64),
            "timestamp_ns": timestamps_ns_arr,
            "time_from_start_s": time_from_start_s,
            "latency_ms": values_ms_arr,
        }
    )


def summarize_latency(metric: str, bag_name: str, df: pd.DataFrame) -> Dict[str, float]:
    x = df["latency_ms"].to_numpy()
    ts = df["timestamp_ns"].to_numpy()

    avg_interval_s = np.nan
    if len(ts) > 1:
        avg_interval_s = np.mean(np.diff(ts) * 1e-9)

    return {
        "metric": metric,
        "bag": bag_name,
        "count": len(x),
        "mean_ms": np.mean(x),
        "std_ms": np.std(x),
        "median_ms": np.median(x),
        "min_ms": np.min(x),
        "max_ms": np.max(x),
        "p90_ms": np.percentile(x, 90),
        "p95_ms": np.percentile(x, 95),
        "p99_ms": np.percentile(x, 99),
        "fps_from_mean": 1000.0 / np.mean(x),
        "avg_interval_s": avg_interval_s,
    }


def plot_time_series(metric: str, dataframes: Dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for bag_name, df in dataframes.items():
        plt.plot(df["sample_index"], df["latency_ms"], linewidth=1.2, label=format_bag_label(bag_name))
    plt.xlabel("Message count index")
    plt.ylabel("Latency [ms]")
    plt.title(f"BEVFusion {metric}: Time Series by Message Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_time_series_small_multiples(metric: str, dataframes: Dict[str, pd.DataFrame], output_path: Path) -> None:
    labels = list(dataframes.keys())
    n = len(labels)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.8 * nrows), sharex=False, sharey=True)
    if isinstance(axes, np.ndarray):
        axes_arr = axes.flatten()
    else:
        axes_arr = np.array([axes])

    for idx, bag_name in enumerate(labels):
        ax = axes_arr[idx]
        df = dataframes[bag_name]

        # Downsample for readability in dense time-series.
        max_points = 400
        stride = max(1, len(df) // max_points)
        view = df.iloc[::stride].copy()

        window = max(5, len(view) // 30)
        trend = view["latency_ms"].rolling(window=window, min_periods=1).mean()

        ax.plot(view["sample_index"], trend, linewidth=2.0, color="tab:blue")
        ax.set_title(format_bag_label(bag_name))
        ax.set_xlabel("Message count index")
        ax.set_ylabel("Latency [ms]")
        ax.grid(True, alpha=0.35)

    for idx in range(n, len(axes_arr)):
        fig.delaxes(axes_arr[idx])

    fig.suptitle(f"BEVFusion {metric}: Smoothed Trend per Bag", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_series_trend_overlay(metric: str, dataframes: Dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    for bag_name, df in dataframes.items():
        # Downsample then smooth to keep one-figure comparison readable.
        max_points = 450
        stride = max(1, len(df) // max_points)
        view = df.iloc[::stride].copy()
        window = max(5, len(view) // 30)
        trend = view["latency_ms"].rolling(window=window, min_periods=1).mean()

        plt.plot(view["sample_index"], trend, linewidth=2.2, label=format_bag_label(bag_name))

    plt.xlabel("Message count index")
    plt.ylabel("Latency [ms]")
    plt.title(f"BEVFusion {metric}: Smoothed Overlay Comparison")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_histogram(metric: str, dataframes: Dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for bag_name, df in dataframes.items():
        plt.hist(df["latency_ms"], bins=40, alpha=0.5, label=format_bag_label(bag_name))
    plt.xlabel("Latency [ms]")
    plt.ylabel("Count")
    plt.title(f"BEVFusion {metric}: Histogram")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_boxplot(metric: str, dataframes: Dict[str, pd.DataFrame], output_path: Path) -> None:
    labels = sorted(dataframes.keys(), key=lambda name: dataframes[name]["latency_ms"].mean(), reverse=True)
    display_labels = [format_bag_label(name) for name in labels]
    values = [dataframes[name]["latency_ms"].to_numpy() for name in labels]
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, labels=display_labels, showmeans=True)
    plt.ylabel("Latency [ms]")
    plt.title(f"BEVFusion {metric}: Boxplot")
    plt.grid(True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cdf(metric: str, dataframes: Dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for bag_name, df in dataframes.items():
        x = np.sort(df["latency_ms"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, linewidth=2, label=format_bag_label(bag_name))
    plt.xlabel("Latency [ms]")
    plt.ylabel("CDF")
    plt.title(f"BEVFusion {metric}: CDF")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def sanitize_filename(name: str) -> str:
    return name.replace("/", "_")


def format_bag_label(bag_name: str) -> str:
    return bag_name.replace("_slow", "(r=0.1)")


def print_metric_summary(metric: str, summary_df: pd.DataFrame) -> None:
    metric_df = summary_df[summary_df["metric"] == metric].copy()
    metric_df = metric_df.sort_values("mean_ms")
    display_cols = ["bag", "count", "mean_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "fps_from_mean"]

    print(f"\n=== {metric} ===")
    print(metric_df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def analyze_metric(
    metric: str,
    topic: str,
    bag_db3_paths: Dict[str, Path],
    output_dir: Path,
) -> List[Dict[str, float]]:
    metric_dir = output_dir / metric
    metric_dir.mkdir(parents=True, exist_ok=True)

    metric_dataframes: Dict[str, pd.DataFrame] = {}
    metric_summaries: List[Dict[str, float]] = []

    for bag_name, db3_path in bag_db3_paths.items():
        print(f"Reading metric={metric}, bag={bag_name}: {db3_path}")
        df = read_float64_stamped_from_rosbag(db3_path, topic)
        df.insert(0, "metric", metric)
        df.insert(1, "bag", bag_name)
        metric_dataframes[bag_name] = df

        raw_csv = metric_dir / f"{sanitize_filename(bag_name)}_raw.csv"
        df.to_csv(raw_csv, index=False)
        metric_summaries.append(summarize_latency(metric, bag_name, df))

    metric_all_raw = pd.concat(metric_dataframes.values(), ignore_index=True)
    metric_all_raw.to_csv(metric_dir / f"{metric}_all_raw.csv", index=False)

    plot_time_series(metric, metric_dataframes, metric_dir / f"{metric}_time_series.png")
    plot_time_series_trend_overlay(metric, metric_dataframes, metric_dir / f"{metric}_trend_overlay.png")
    plot_time_series_small_multiples(metric, metric_dataframes, metric_dir / f"{metric}_trend_small_multiples.png")
    plot_histogram(metric, metric_dataframes, metric_dir / f"{metric}_histogram.png")
    plot_boxplot(metric, metric_dataframes, metric_dir / f"{metric}_boxplot.png")
    plot_cdf(metric, metric_dataframes, metric_dir / f"{metric}_cdf.png")

    print(f"Saved metric outputs to: {metric_dir}")
    return metric_summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bag_dirs = discover_bag_dirs(args.rosbag_root)
    if args.bags:
        bag_dirs = resolve_requested_bags(bag_dirs, args.bags)

    bag_db3_paths: Dict[str, Path] = {}
    for bag_name, bag_dir in bag_dirs.items():
        bag_db3_paths[bag_name] = resolve_db3_path(bag_dir)

    print("Resolved bag files:")
    for bag_name, db3_path in bag_db3_paths.items():
        print(f"- {bag_name}: {db3_path}")

    all_summaries: List[Dict[str, float]] = []

    rclpy.init()
    try:
        for metric, topic in METRIC_TOPICS.items():
            all_summaries.extend(analyze_metric(metric, topic, bag_db3_paths, args.output_dir))
    finally:
        rclpy.shutdown()

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = args.output_dir / "summary_all_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    for metric in METRIC_TOPICS.keys():
        print_metric_summary(metric, summary_df)


if __name__ == "__main__":
    main()

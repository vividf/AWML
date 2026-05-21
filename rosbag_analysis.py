#!/usr/bin/env python3

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rclpy
from autoware_internal_debug_msgs.msg import Float64Stamped
from rclpy.serialization import deserialize_message

TOPIC = "/perception/object_recognition/detection/centerpoint/lidar_centerpoint/debug/processing_time_ms"

BAGS = {
    "FP16": "/media/yihsiangfang/VIVID/centerpoint_2_6/rosbag2_fp16_test2/rosbag2_2026_05_18-18_35_47_0.db3",
    "INT8": "/media/yihsiangfang/VIVID/centerpoint_2_6/rosbag2_int8_test2/rosbag2_2026_05_18-18_42_30_0.db3",
}

OUTPUT_DIR = Path("./centerpoint_latency_analysis")


def read_float64_stamped_from_rosbag(db3_path: str, topic_name: str) -> pd.DataFrame:
    db3_path = Path(db3_path)

    if not db3_path.exists():
        raise FileNotFoundError(f"Bag file does not exist: {db3_path}")

    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM topics WHERE name = ?", (topic_name,))
    row = cur.fetchone()

    if row is None:
        available_topics = cur.execute("SELECT name FROM topics").fetchall()
        conn.close()
        available_topics = [t[0] for t in available_topics]
        raise RuntimeError(f"Topic not found: {topic_name}\n" f"Available topics:\n" + "\n".join(available_topics))

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

    if len(values_ms) == 0:
        raise RuntimeError(f"No messages found for topic: {topic_name}")

    timestamps_ns = np.array(timestamps_ns, dtype=np.int64)
    values_ms = np.array(values_ms, dtype=np.float64)

    time_from_start_s = (timestamps_ns - timestamps_ns[0]) * 1e-9

    return pd.DataFrame(
        {
            "timestamp_ns": timestamps_ns,
            "time_from_start_s": time_from_start_s,
            "latency_ms": values_ms,
        }
    )


def summarize_latency(name: str, df: pd.DataFrame) -> dict:
    x = df["latency_ms"].to_numpy()

    return {
        "name": name,
        "count": len(x),
        "mean_ms": np.mean(x),
        "std_ms": np.std(x),
        "median_ms": np.median(x),
        "min_ms": np.min(x),
        "max_ms": np.max(x),
        "p50_ms": np.percentile(x, 50),
        "p90_ms": np.percentile(x, 90),
        "p95_ms": np.percentile(x, 95),
        "p99_ms": np.percentile(x, 99),
        "fps_from_mean": 1000.0 / np.mean(x),
        "avg_interval_s": np.mean(np.diff(df["timestamp_ns"].to_numpy()) * 1e-9),
    }


def print_summary(summary_df: pd.DataFrame) -> None:
    display_cols = [
        "name",
        "count",
        "mean_ms",
        "std_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "fps_from_mean",
        "avg_interval_s",
    ]

    print("\n=== Latency Summary ===")
    print(summary_df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if {"FP16", "INT8"}.issubset(set(summary_df["name"])):
        fp16_mean = summary_df.loc[summary_df["name"] == "FP16", "mean_ms"].iloc[0]
        int8_mean = summary_df.loc[summary_df["name"] == "INT8", "mean_ms"].iloc[0]

        fp16_p95 = summary_df.loc[summary_df["name"] == "FP16", "p95_ms"].iloc[0]
        int8_p95 = summary_df.loc[summary_df["name"] == "INT8", "p95_ms"].iloc[0]

        fp16_p99 = summary_df.loc[summary_df["name"] == "FP16", "p99_ms"].iloc[0]
        int8_p99 = summary_df.loc[summary_df["name"] == "INT8", "p99_ms"].iloc[0]

        mean_improvement = (fp16_mean - int8_mean) / fp16_mean * 100.0
        p95_improvement = (fp16_p95 - int8_p95) / fp16_p95 * 100.0
        p99_improvement = (fp16_p99 - int8_p99) / fp16_p99 * 100.0

        print("\n=== Improvement: INT8 vs FP16 ===")
        print(f"Mean latency improvement : {mean_improvement:.2f}%")
        print(f"P95 latency improvement  : {p95_improvement:.2f}%")
        print(f"P99 latency improvement  : {p99_improvement:.2f}%")


def plot_time_series(dataframes: dict, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    for name, df in dataframes.items():
        plt.plot(df["time_from_start_s"], df["latency_ms"], label=name, linewidth=1.2)

    plt.xlabel("Time from start [s]")
    plt.ylabel("Processing time [ms]")
    plt.title("CenterPoint Processing Time: Time Series")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_histogram(dataframes: dict, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    for name, df in dataframes.items():
        plt.hist(df["latency_ms"], bins=40, alpha=0.5, label=name)

    plt.xlabel("Processing time [ms]")
    plt.ylabel("Count")
    plt.title("CenterPoint Processing Time: Histogram")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_boxplot(dataframes: dict, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))

    labels = list(dataframes.keys())
    values = [dataframes[name]["latency_ms"].to_numpy() for name in labels]

    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("Processing time [ms]")
    plt.title("CenterPoint Processing Time: Boxplot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cdf(dataframes: dict, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    for name, df in dataframes.items():
        x = np.sort(df["latency_ms"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=name, linewidth=2)

    plt.xlabel("Processing time [ms]")
    plt.ylabel("CDF")
    plt.title("CenterPoint Processing Time: CDF")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    rclpy.init()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframes = {}

    for name, bag_path in BAGS.items():
        print(f"Reading {name}: {bag_path}")
        df = read_float64_stamped_from_rosbag(bag_path, TOPIC)
        df.insert(0, "mode", name)
        dataframes[name] = df

        raw_csv_path = OUTPUT_DIR / f"{name.lower()}_processing_time_raw.csv"
        df.to_csv(raw_csv_path, index=False)
        print(f"Saved raw data: {raw_csv_path}")

    summaries = [summarize_latency(name, df) for name, df in dataframes.items()]
    summary_df = pd.DataFrame(summaries)

    summary_csv_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print_summary(summary_df)
    print(f"\nSaved summary: {summary_csv_path}")

    all_raw_df = pd.concat(dataframes.values(), ignore_index=True)
    all_raw_csv_path = OUTPUT_DIR / "all_processing_time_raw.csv"
    all_raw_df.to_csv(all_raw_csv_path, index=False)
    print(f"Saved combined raw data: {all_raw_csv_path}")

    plot_time_series(dataframes, OUTPUT_DIR / "time_series.png")
    plot_histogram(dataframes, OUTPUT_DIR / "histogram.png")
    plot_boxplot(dataframes, OUTPUT_DIR / "boxplot.png")
    plot_cdf(dataframes, OUTPUT_DIR / "cdf.png")

    print("\nSaved plots:")
    print(f"- {OUTPUT_DIR / 'time_series.png'}")
    print(f"- {OUTPUT_DIR / 'histogram.png'}")
    print(f"- {OUTPUT_DIR / 'boxplot.png'}")
    print(f"- {OUTPUT_DIR / 'cdf.png'}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()

#-*- coding: utf-8 -*-
"""
rewardAnalysis.py

Read reward_breakdown_swXXX.csv and plot selected metrics vs time.
Each figure:
    x-axis: time_sec
    y-axis: one metric

Usage example:
    python rewardAnalysis.py \
        --csv reward_breakdown_sw128.csv \
        --metrics T qSmooth r_level \
        --out-dir figs

    # plot all metrics (except time_sec, out_if)
    python rewardAnalysis.py \
        --csv reward_breakdown_sw128.csv \
        --metrics all
"""

import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot reward components over time from reward_breakdown_swXXX.csv"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the reward_breakdown_swXXX.csv file"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["T", "qSmooth", "R_norm", "R_qcn", "r_inst", "r_level", "delta"],
        help=("which columns to plot."
              "Use 'all' to plot all metrics (except time_sec, out_if)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="reward_figs",
        help="Directory to save the figures(will be created if not exists).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for the saved figures.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png","pdf"],
        help="Format for the saved figures.",
    )
    return parser.parse_args()

def infer_switch_id(csv_path: str) -> str:
    """Try to infer switch ID from filename like reward_breakdown_sw128.csv"""
    fname = os.path.basename(csv_path)
    m =  re.search(r"sw(\d+)", fname)
    if m:
        return m.group(1)
    return "unknown"

def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)

    required_cols = {"time_sec", "out_if"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}, got: {list(df.columns)}")

    if len(args.metrics) == 1 and args.metrics[0] == "all":
        metrics = [c for c in df.columns if c not in ("time_sec", "out_if")]
    else:
        metrics = args.metrics

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print(
            f"[WARN] These metrics are not found in the CSV: {missing}. Skipping..."
        )
        metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        raise ValueError("No valid metrics to plot after filtering.")

    os.makedirs(args.out_dir, exist_ok=True)

    sw_id = infer_switch_id(args.csv)

    grouped = df.groupby("out_if")

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))

        for out_if, sub in grouped:
            sub_sorted = sub.sort_values("time_sec")
            ax.plot(
                sub_sorted["time_sec"],
                sub_sorted[metric],
                label=f"out_if {out_if}",
                linewidth=1.0,
            )

        ax.set_xlabel("time_sec")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} over time (sw {sw_id})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

        out_fname = f"sw{sw_id}_{metric}.{args.format}"
        out_path = os.path.join(args.out_dir, out_fname)
        plt.tight_layout()
        fig.savefig(out_path, dpi=args.dpi, format=args.format)
        plt.close(fig)

        print(f"[INFO] Saved figure to: {out_path}")
    print(f"[INFO] All selected metrics have been plotted.")


if __name__ == "__main__":
    main()
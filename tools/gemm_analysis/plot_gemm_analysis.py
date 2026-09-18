#!/usr/bin/env python3
"""
GEMM analysis plot generator.

Inputs:
  - gemm_analysis_runs.csv
  - gemm_analysis_summary.json

Outputs (in same directory unless overridden):
  - roofline.png
  - scaling.png
  - cache_transitions.png
  - simd_utilization.png
  - stability.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_inputs(csv_path: Path, summary_path: Path):
    df = pd.read_csv(csv_path)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return df, summary


def safe_log_values(values, floor=1e-6):
    return np.clip(np.asarray(values, dtype=float), floor, None)


def plot_roofline(df: pd.DataFrame, summary: dict, out: Path):
    data = df[(df["scenario"] == "geometry") & (df["simd_enabled"] == True)].copy()
    if data.empty:
        return

    x = safe_log_values(data["arithmetic_intensity_modeled"])
    y = safe_log_values(data["measured_gflops_best"])

    compute_roof = float(summary["roofline"]["compute_roof_gflops"])
    bw_levels = {
        "L1": float(summary["roofline"]["memory_l1_gbps"]),
        "L2": float(summary["roofline"]["memory_l2_gbps"]),
        "L3": float(summary["roofline"]["memory_l3_gbps"]),
        "DRAM": float(summary["roofline"]["memory_dram_gbps"]),
    }

    ai_min = max(1e-3, float(np.min(x)) * 0.8)
    ai_max = max(1.0, float(np.max(x)) * 1.3)
    ai_grid = np.logspace(np.log10(ai_min), np.log10(ai_max), 400)

    fig, ax = plt.subplots(figsize=(12, 8))

    regime_colors = {"l1": "#1b9e77", "l2": "#7570b3", "l3": "#d95f02", "dram": "#e7298a"}
    colors = data["cache_regime"].map(lambda r: regime_colors.get(str(r).lower(), "#666666"))

    sc = ax.scatter(x, y, c=colors, s=42, alpha=0.85, edgecolor="black", linewidths=0.35)

    for level, bw in bw_levels.items():
        roof_line = np.minimum(compute_roof, bw * ai_grid)
        ax.plot(ai_grid, roof_line, linewidth=1.8, label=f"{level} roof ({bw:.1f} GB/s)")

    ax.axhline(compute_roof, color="black", linestyle="--", linewidth=1.7, label=f"Compute roof ({compute_roof:.1f} GF/s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Measured Throughput (GFLOP/s)")
    ax.set_title("GEMM Roofline Placement")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black", markersize=8, label=reg)
        for reg, c in regime_colors.items()
    ]
    ax.legend(handles=legend_handles, title="Cache regime", loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_scaling(df: pd.DataFrame, out: Path):
    data = df[(df["scenario"] == "scaling") & (df["simd_enabled"] == True)].copy()
    if data.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for case_id, g in data.groupby("case_id"):
        g = g.sort_values("threads")
        ax1.plot(g["threads"], g["speedup_vs_1thread"], marker="o", linewidth=2, label=case_id)
        ax2.plot(g["threads"], g["parallel_efficiency"], marker="o", linewidth=2, label=case_id)

    max_threads = int(data["threads"].max())
    ax1.plot([1, max_threads], [1, max_threads], "k--", alpha=0.55, label="ideal")

    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Speedup vs 1 thread")
    ax1.set_title("Strong Scaling")
    ax1.grid(True, alpha=0.25)

    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.55)
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Parallel Efficiency")
    ax2.set_title("Scaling Efficiency")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.25)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_cache_transitions(df: pd.DataFrame, out: Path):
    data = df[
        (df["scenario"] == "geometry")
        & (df["shape"] == "square")
        & (df["simd_enabled"] == True)
    ].copy()
    if data.empty:
        return

    max_threads = int(data["threads"].max())
    data = data[data["threads"] == max_threads].sort_values("working_set_bytes")

    fig, ax = plt.subplots(figsize=(11, 6))
    x_mb = data["working_set_bytes"] / (1024.0 * 1024.0)
    y = data["measured_gflops_best"]

    regime_colors = {"l1": "#1b9e77", "l2": "#7570b3", "l3": "#d95f02", "dram": "#e7298a"}
    colors = data["cache_regime"].map(lambda r: regime_colors.get(str(r).lower(), "#666666"))

    ax.plot(x_mb, y, color="#1f77b4", linewidth=1.5, alpha=0.6)
    ax.scatter(x_mb, y, c=colors, s=48, edgecolor="black", linewidths=0.35)

    for _, row in data.iterrows():
        if row["cache_regime"] != data.iloc[0]["cache_regime"]:
            break

    ax.set_xscale("log")
    ax.set_xlabel("Working-set size (MiB)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Cache-Regime Transitions (square, threads={max_threads})")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_simd(df: pd.DataFrame, out: Path):
    on = df[(df["scenario"] == "simd_probe") & (df["variant"] == "simd_on")].copy()
    if on.empty:
        return

    on = on.sort_values("n")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(on["n"], on["simd_speedup_vs_scalar"], marker="o", linewidth=2, color="#2c7fb8")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Matrix size n (square n x n x n)")
    ax.set_ylabel("SIMD speedup vs scalar")
    ax.set_title("SIMD Utilization Probe")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_stability(df: pd.DataFrame, out: Path):
    data = df[(df["scenario"] == "geometry") & (df["simd_enabled"] == True)].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = data["working_set_bytes"] / (1024.0 * 1024.0)
    y = data["cv"]

    ax.scatter(x, y, c=data["threads"], cmap="viridis", s=50, edgecolor="black", linewidths=0.3)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.3, label="CV=5%")

    ax.set_xscale("log")
    ax.set_xlabel("Working-set size (MiB)")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Performance Stability")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate GEMM analysis plots")
    parser.add_argument("--csv", default="build/reports/gemm-analysis/gemm_analysis_runs.csv", help="Path to runs CSV")
    parser.add_argument("--summary", default="build/reports/gemm-analysis/gemm_analysis_summary.json", help="Path to summary JSON")
    parser.add_argument("--out-dir", default="build/reports/gemm-analysis", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, summary = load_inputs(csv_path, summary_path)

    plot_roofline(df, summary, out_dir / "roofline.png")
    plot_scaling(df, out_dir / "scaling.png")
    plot_cache_transitions(df, out_dir / "cache_transitions.png")
    plot_simd(df, out_dir / "simd_utilization.png")
    plot_stability(df, out_dir / "stability.png")

    print("Generated:")
    for f in ["roofline.png", "scaling.png", "cache_transitions.png", "simd_utilization.png", "stability.png"]:
        p = out_dir / f
        if p.exists():
            print(f"  - {p}")


if __name__ == "__main__":
    main()

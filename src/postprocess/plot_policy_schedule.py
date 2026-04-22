#!/usr/bin/env python3
"""Plot policy decision-variable schedules in a paper-style layout.

This script reads the long-format schedule CSV produced by run/multi_dc.py:
    sample_id,var_name,index,value

It creates:
1) A multi-panel time-series figure for all time-indexed decision variables.
2) A bar figure for non-time-indexed decision variables (if any).

Usage:
    python src/postprocess/plot_policy_schedule.py \
        --input result/mdc_policy_schedule_cls55.0_T288.csv \
        --output-dir result/figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BINARY_VARS = {
    "u_th",
    "y",
    "on",
    "xI",
    "z",
    "delta",
    "mI",
    "mB",
}

# Variables that do not use time as index in this formulation.
NON_TIME_VARS = {
    "delta",
}


def _configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.6,
        }
    )


def _find_latest_schedule_csv(result_dir: Path) -> Path:
    candidates = sorted(result_dir.glob("mdc_policy_schedule_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No schedule CSV found under: {result_dir}")
    return candidates[0]


def _parse_time_index(var_name: str, index_text: str) -> Optional[int]:
    if var_name in NON_TIME_VARS:
        return None

    if index_text is None:
        return None

    s = str(index_text).strip()
    if s == "":
        return None

    parts = s.split("|")
    last = parts[-1]
    try:
        return int(last)
    except ValueError:
        return None


def _subplot_grid(n: int) -> tuple[int, int]:
    cols = 4 if n >= 8 else 3 if n >= 5 else 2 if n >= 2 else 1
    rows = int(math.ceil(n / cols))
    return rows, cols


def _split_index_to_ints(index_text: str) -> Optional[list[int]]:
    if index_text is None:
        return None
    s = str(index_text).strip()
    if s == "":
        return None
    parts = s.split("|")
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            return None
    return out


def _extract_var_dims(df: pd.DataFrame, var_name: str, dim_names: list[str]) -> pd.DataFrame:
    sub = df[df["var_name"] == var_name][["sample_id", "index", "value"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["sample_id", *dim_names, "value"])

    parsed = sub["index"].astype(str).map(_split_index_to_ints)
    lens = parsed.map(lambda x: len(x) if x is not None else -1)
    sub = sub[lens == len(dim_names)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["sample_id", *dim_names, "value"])

    parsed = sub["index"].astype(str).map(_split_index_to_ints)
    for i, name in enumerate(dim_names):
        sub[name] = parsed.map(lambda x, ii=i: x[ii] if x is not None else np.nan)

    keep_cols = ["sample_id", *dim_names, "value"]
    return sub[keep_cols]


def _plot_time_variables(df: pd.DataFrame, out_dir: Path, stem: str) -> Optional[Path]:
    df = df.copy()
    df["t"] = [
        _parse_time_index(vn, idx)
        for vn, idx in zip(df["var_name"].astype(str).values, df["index"].astype(str).values)
    ]
    dft = df[df["t"].notna()].copy()
    if dft.empty:
        return None

    dft["t"] = dft["t"].astype(int)
    agg = (
        dft.groupby(["var_name", "t"], as_index=False)["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    var_names = sorted(agg["var_name"].unique().tolist())
    rows, cols = _subplot_grid(len(var_names))

    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, var_name in enumerate(var_names):
        ax = axes_flat[i]
        sub = agg[agg["var_name"] == var_name].sort_values("t")
        t = sub["t"].to_numpy()
        y = sub["mean"].to_numpy(dtype=float)
        s = np.nan_to_num(sub["std"].to_numpy(dtype=float), nan=0.0)

        color = "#1f77b4" if var_name not in BINARY_VARS else "#d62728"
        ax.plot(t, y, color=color, label="mean")
        ax.fill_between(t, y - s, y + s, color=color, alpha=0.18, linewidth=0.0, label="mean ± std")

        ax.set_title(var_name)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")

        if var_name in BINARY_VARS:
            ax.set_ylim(-0.05, 1.05)

        if i == 0:
            ax.legend(frameon=False, loc="best")

    for j in range(len(var_names), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Decision Variables Over Scheduling Horizon", y=1.01, fontsize=13)
    out_path = out_dir / f"{stem}_timeseries.pdf"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}_timeseries.png")
    plt.close(fig)
    return out_path


def _plot_non_time_variables(df: pd.DataFrame, out_dir: Path, stem: str) -> Optional[Path]:
    df = df.copy()
    df["t"] = [
        _parse_time_index(vn, idx)
        for vn, idx in zip(df["var_name"].astype(str).values, df["index"].astype(str).values)
    ]
    dfs = df[df["t"].isna()].copy()
    if dfs.empty:
        return None

    # For non-time variables, summarize by variable and index across samples.
    grp = (
        dfs.groupby(["var_name", "index"], as_index=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    var_names = sorted(grp["var_name"].unique().tolist())
    rows, cols = _subplot_grid(len(var_names))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, var_name in enumerate(var_names):
        ax = axes_flat[i]
        sub = grp[grp["var_name"] == var_name].copy()
        sub["idx_num"] = pd.to_numeric(sub["index"], errors="coerce")
        sub = sub.sort_values("idx_num", na_position="last")

        x = np.arange(len(sub))
        y = sub["mean"].to_numpy(dtype=float)
        s = np.nan_to_num(sub["std"].to_numpy(dtype=float), nan=0.0)

        color = "#2ca02c" if var_name not in BINARY_VARS else "#ff7f0e"
        ax.bar(x, y, color=color, alpha=0.85, width=0.75)
        ax.errorbar(x, y, yerr=s, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)

        ax.set_title(var_name)
        ax.set_xlabel("Index")
        ax.set_ylabel("Mean value")

        if var_name in BINARY_VARS:
            ax.set_ylim(-0.05, 1.05)

        # Thin tick labels for readability when many indices exist.
        if len(x) > 18:
            step = max(1, len(x) // 12)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([str(v) for v in sub["index"].astype(str).iloc[::step]], rotation=0)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in sub["index"].astype(str)], rotation=0)

    for j in range(len(var_names), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Non-time Decision Variables", y=1.01, fontsize=13)
    out_path = out_dir / f"{stem}_static.pdf"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}_static.png")
    plt.close(fig)
    return out_path


def _plot_dc_workload_processing(df: pd.DataFrame, out_dir: Path, stem: str) -> Optional[Path]:
    # Interactive workload proxy from xI assignment frequency/intensity.
    xI = _extract_var_dims(df, "xI", ["dc", "inode", "t"])
    xI_stat = None
    if not xI.empty:
        xI_per_sample = xI.groupby(["sample_id", "dc", "t"], as_index=False)["value"].sum()
        xI_stat = (
            xI_per_sample.groupby(["dc", "t"], as_index=False)["value"]
            .agg(["mean", "std"]) 
            .reset_index()
        )

    # Batch workload proxy from f (processed batch amount).
    f = _extract_var_dims(df, "f", ["dc", "bnode", "t"])
    f_stat = None
    if not f.empty:
        f_per_sample = f.groupby(["sample_id", "dc", "t"], as_index=False)["value"].sum()
        f_stat = (
            f_per_sample.groupby(["dc", "t"], as_index=False)["value"]
            .agg(["mean", "std"]) 
            .reset_index()
        )

    available = [s for s in [xI_stat, f_stat] if s is not None and not s.empty]
    if not available:
        return None

    dc_ids = sorted(
        set(int(v) for s in available for v in s["dc"].dropna().astype(int).unique().tolist())
    )
    if not dc_ids:
        return None

    fig, axes = plt.subplots(2, len(dc_ids), figsize=(3.8 * len(dc_ids), 5.4), squeeze=False, sharex=False)

    for col, dc in enumerate(dc_ids):
        ax_i = axes[0, col]
        ax_b = axes[1, col]

        if xI_stat is not None and not xI_stat.empty:
            sub = xI_stat[xI_stat["dc"] == dc].sort_values("t")
            t = sub["t"].to_numpy(dtype=int)
            y = sub["mean"].to_numpy(dtype=float)
            s = np.nan_to_num(sub["std"].to_numpy(dtype=float), nan=0.0)
            ax_i.plot(t, y, color="#1f77b4")
            ax_i.fill_between(t, y - s, y + s, color="#1f77b4", alpha=0.18, linewidth=0.0)
        ax_i.set_title(f"DC {dc} - Interactive")
        ax_i.set_xlabel("Time step")
        ax_i.set_ylabel("Processed workload")

        if f_stat is not None and not f_stat.empty:
            sub = f_stat[f_stat["dc"] == dc].sort_values("t")
            t = sub["t"].to_numpy(dtype=int)
            y = sub["mean"].to_numpy(dtype=float)
            s = np.nan_to_num(sub["std"].to_numpy(dtype=float), nan=0.0)
            ax_b.plot(t, y, color="#2ca02c")
            ax_b.fill_between(t, y - s, y + s, color="#2ca02c", alpha=0.18, linewidth=0.0)
        ax_b.set_title(f"DC {dc} - Batch")
        ax_b.set_xlabel("Time step")
        ax_b.set_ylabel("Processed workload")

    fig.suptitle("Per-DC Workload Processing", y=1.01, fontsize=13)
    out_path = out_dir / f"{stem}_dc_workload.pdf"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}_dc_workload.png")
    plt.close(fig)
    return out_path


def _plot_migration_journal(df: pd.DataFrame, out_dir: Path, stem: str) -> Optional[Path]:
    mI = _extract_var_dims(df, "mI", ["wnode", "src", "dst", "t"])
    mB = _extract_var_dims(df, "mB", ["wnode", "src", "dst", "t"])

    if mI.empty and mB.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), squeeze=False)
    ax_ts = axes[0, 0]
    ax_hm = axes[0, 1]

    pair_matrix = None

    if not mI.empty:
        ts = mI.groupby(["sample_id", "t"], as_index=False)["value"].sum()
        ts = ts.groupby("t", as_index=False)["value"].mean().sort_values("t")
        ax_ts.plot(ts["t"].to_numpy(dtype=int), ts["value"].to_numpy(dtype=float), label="mI", color="#d62728")

        pair = mI.groupby(["sample_id", "src", "dst"], as_index=False)["value"].sum()
        pair = pair.groupby(["src", "dst"], as_index=False)["value"].mean()
        pair["kind"] = "mI"
        pair_matrix = pair

    if not mB.empty:
        ts = mB.groupby(["sample_id", "t"], as_index=False)["value"].sum()
        ts = ts.groupby("t", as_index=False)["value"].mean().sort_values("t")
        ax_ts.plot(ts["t"].to_numpy(dtype=int), ts["value"].to_numpy(dtype=float), label="mB", color="#9467bd")

        pair = mB.groupby(["sample_id", "src", "dst"], as_index=False)["value"].sum()
        pair = pair.groupby(["src", "dst"], as_index=False)["value"].mean()
        pair["kind"] = "mB"
        if pair_matrix is None:
            pair_matrix = pair
        else:
            pair_matrix = pd.concat([pair_matrix, pair], ignore_index=True)

    ax_ts.set_title("Migration Timeline")
    ax_ts.set_xlabel("Time step")
    ax_ts.set_ylabel("Mean migrated load")
    ax_ts.legend(frameon=False, loc="best")

    if pair_matrix is None or pair_matrix.empty:
        ax_hm.axis("off")
    else:
        pair_sum = pair_matrix.groupby(["src", "dst"], as_index=False)["value"].sum()
        src_ids = sorted(pair_sum["src"].astype(int).unique().tolist())
        dst_ids = sorted(pair_sum["dst"].astype(int).unique().tolist())
        mat = np.zeros((len(src_ids), len(dst_ids)), dtype=float)
        src_pos = {v: i for i, v in enumerate(src_ids)}
        dst_pos = {v: i for i, v in enumerate(dst_ids)}
        for _, r in pair_sum.iterrows():
            mat[src_pos[int(r["src"])] , dst_pos[int(r["dst"])]] = float(r["value"])

        im = ax_hm.imshow(mat, aspect="auto", cmap="YlOrRd", origin="lower")
        ax_hm.set_title("Migration Pair Intensity")
        ax_hm.set_xlabel("Destination DC")
        ax_hm.set_ylabel("Source DC")
        ax_hm.set_xticks(np.arange(len(dst_ids)))
        ax_hm.set_xticklabels([str(v) for v in dst_ids])
        ax_hm.set_yticks(np.arange(len(src_ids)))
        ax_hm.set_yticklabels([str(v) for v in src_ids])
        fig.colorbar(im, ax=ax_hm, shrink=0.9, label="Mean migrated load")

    fig.suptitle("Cross-DC Migration Journal", y=1.02, fontsize=13)
    out_path = out_dir / f"{stem}_migration_journal.pdf"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}_migration_journal.png")
    plt.close(fig)
    return out_path


def _plot_worker_node_frequency_heatmaps(df: pd.DataFrame, out_dir: Path, stem: str) -> Optional[Path]:
    def build_matrix(var_name: str) -> tuple[np.ndarray, list[int], list[int]]:
        var_df = _extract_var_dims(df, var_name, ["dc", "node", "t"])
        if var_df.empty:
            return np.zeros((0, 0), dtype=float), [], []

        freq = (
            var_df.assign(active=(var_df["value"] > 1e-9).astype(float))
            .groupby(["dc", "node"], as_index=False)["active"]
            .mean()
        )
        dcs = sorted(freq["dc"].astype(int).unique().tolist())
        nodes = sorted(freq["node"].astype(int).unique().tolist())
        mat = np.zeros((len(nodes), len(dcs)), dtype=float)
        dc_pos = {v: i for i, v in enumerate(dcs)}
        node_pos = {v: i for i, v in enumerate(nodes)}
        for _, r in freq.iterrows():
            mat[node_pos[int(r["node"])], dc_pos[int(r["dc"])]] = float(r["active"])
        return mat, nodes, dcs

    mat_xI, nodes_xI, dcs_xI = build_matrix("xI")
    mat_z, nodes_z, dcs_z = build_matrix("z")

    if mat_xI.size == 0 and mat_z.size == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), squeeze=False)

    ax1 = axes[0, 0]
    if mat_xI.size > 0:
        im1 = ax1.imshow(mat_xI, aspect="auto", cmap="Blues", origin="lower", vmin=0.0, vmax=1.0)
        ax1.set_title("xI Worker-Node Activation Frequency")
        ax1.set_xlabel("Data Center")
        ax1.set_ylabel("Worker Node")
        ax1.set_xticks(np.arange(len(dcs_xI)))
        ax1.set_xticklabels([str(v) for v in dcs_xI])
        ax1.set_yticks(np.arange(len(nodes_xI)))
        if len(nodes_xI) > 24:
            step = max(1, len(nodes_xI) // 12)
            ax1.set_yticks(np.arange(0, len(nodes_xI), step))
            ax1.set_yticklabels([str(nodes_xI[i]) for i in range(0, len(nodes_xI), step)])
        else:
            ax1.set_yticklabels([str(v) for v in nodes_xI])
        fig.colorbar(im1, ax=ax1, shrink=0.9, label="Activation frequency")
    else:
        ax1.axis("off")

    ax2 = axes[0, 1]
    if mat_z.size > 0:
        im2 = ax2.imshow(mat_z, aspect="auto", cmap="Greens", origin="lower", vmin=0.0, vmax=1.0)
        ax2.set_title("z Worker-Node Activation Frequency")
        ax2.set_xlabel("Data Center")
        ax2.set_ylabel("Worker Node")
        ax2.set_xticks(np.arange(len(dcs_z)))
        ax2.set_xticklabels([str(v) for v in dcs_z])
        ax2.set_yticks(np.arange(len(nodes_z)))
        if len(nodes_z) > 24:
            step = max(1, len(nodes_z) // 12)
            ax2.set_yticks(np.arange(0, len(nodes_z), step))
            ax2.set_yticklabels([str(nodes_z[i]) for i in range(0, len(nodes_z), step)])
        else:
            ax2.set_yticklabels([str(v) for v in nodes_z])
        fig.colorbar(im2, ax=ax2, shrink=0.9, label="Activation frequency")
    else:
        ax2.axis("off")

    fig.suptitle("Worker-Node Frequency Heatmaps", y=1.02, fontsize=13)
    out_path = out_dir / f"{stem}_worker_node_frequency_heatmap.pdf"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{stem}_worker_node_frequency_heatmap.png")
    plt.close(fig)
    return out_path


def _read_csv(path: Path, sample_limit: Optional[int]) -> pd.DataFrame:
    usecols = ["sample_id", "var_name", "index", "value"]
    df = pd.read_csv(path, usecols=usecols)

    if sample_limit is not None and sample_limit > 0:
        keep_ids = sorted(df["sample_id"].dropna().unique().tolist())[:sample_limit]
        df = df[df["sample_id"].isin(keep_ids)].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])  # keep plotting robust on malformed rows
    return df


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot decision-variable schedules from policy CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to mdc_policy_schedule_*.csv. If omitted, uses the newest file under result/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional: only use the first N sample_id values for faster plotting.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    _configure_paper_style()

    repo_root = Path(__file__).resolve().parents[2]
    result_dir = repo_root / "result"

    if args.input:
        csv_path = Path(args.input)
        if not csv_path.is_absolute():
            csv_path = (repo_root / csv_path).resolve()
    else:
        csv_path = _find_latest_schedule_csv(result_dir)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    print(f"[Plot] Loading: {csv_path}")
    df = _read_csv(csv_path, sample_limit=args.sample_limit)
    print(f"[Plot] Rows used: {len(df)}")

    stem = csv_path.stem
    t_fig = _plot_time_variables(df, out_dir, stem)
    s_fig = _plot_non_time_variables(df, out_dir, stem)
    dc_fig = _plot_dc_workload_processing(df, out_dir, stem)
    mig_fig = _plot_migration_journal(df, out_dir, stem)
    heat_fig = _plot_worker_node_frequency_heatmaps(df, out_dir, stem)

    if t_fig is not None:
        print(f"[Plot] Saved time-series figure: {t_fig}")
    else:
        print("[Plot] No time-indexed variables found.")

    if s_fig is not None:
        print(f"[Plot] Saved static-variable figure: {s_fig}")
    else:
        print("[Plot] No non-time variables found.")

    if dc_fig is not None:
        print(f"[Plot] Saved per-DC workload figure: {dc_fig}")
    else:
        print("[Plot] No workload variables (xI/f) found.")

    if mig_fig is not None:
        print(f"[Plot] Saved migration-journal figure: {mig_fig}")
    else:
        print("[Plot] No migration variables (mI/mB) found.")

    if heat_fig is not None:
        print(f"[Plot] Saved worker-node frequency heatmap: {heat_fig}")
    else:
        print("[Plot] No worker-node variables (xI/z) found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

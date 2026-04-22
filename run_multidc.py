#!/usr/bin/env python
# coding: utf-8
"""
Run script for Multi-DC scheduling with fixed 24h@5min setup.

This script intentionally uses plain-text hardcoded profiles in code
to keep all fixed parameters and source data transparent and reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import run
from src.utlis import data_split


FIXED_TIME_STEP_HOURS = 5.0 / 60.0
FIXED_HORIZON = 24 * 12
FIXED_NUM_DC = 3
FIXED_NUM_REGIONS = 3
FIXED_NUM_JOBS = 4
FIXED_SOLVER = "gurobi_persistent"


# Hourly anchors (24 points) written in plain text.
# Units:
# - load_mw: feeder non-IT base demand (MW)
# - price_usd_per_mwh: tariff (USD/MWh)
# - pv_norm / wind_norm: normalized renewable availability
_LOAD_MW_HOURLY = np.array(
    [
        1.85, 1.78, 1.73, 1.70, 1.72, 1.80,
        1.98, 2.18, 2.35, 2.46, 2.55, 2.62,
        2.58, 2.50, 2.45, 2.52, 2.66, 2.84,
        3.00, 3.08, 2.96, 2.70, 2.35, 2.02,
    ],
    dtype=np.float32,
)
_PRICE_USD_PER_MWH_HOURLY = np.array(
    [
        62.0, 58.0, 55.0, 53.0, 54.0, 60.0,
        72.0, 88.0, 96.0, 102.0, 106.0, 110.0,
        104.0, 98.0, 94.0, 101.0, 116.0, 132.0,
        145.0, 152.0, 138.0, 112.0, 88.0, 72.0,
    ],
    dtype=np.float32,
)
_PV_NORM_HOURLY = np.array(
    [
        0.00, 0.00, 0.00, 0.00, 0.01, 0.05,
        0.15, 0.30, 0.48, 0.66, 0.80, 0.90,
        0.95, 0.92, 0.82, 0.66, 0.48, 0.28,
        0.12, 0.03, 0.00, 0.00, 0.00, 0.00,
    ],
    dtype=np.float32,
)
_WIND_NORM_HOURLY = np.array(
    [
        0.62, 0.58, 0.55, 0.53, 0.52, 0.50,
        0.48, 0.45, 0.42, 0.40, 0.38, 0.36,
        0.34, 0.35, 0.38, 0.42, 0.48, 0.54,
        0.60, 0.64, 0.66, 0.67, 0.66, 0.64,
    ],
    dtype=np.float32,
)

# Offline "real-like" profile anchors (24 hourly points) for reproducible no-network runs.
_REAL_LOAD_MW_HOURLY = np.array(
    [
        1.62, 1.56, 1.52, 1.50, 1.54, 1.66,
        1.92, 2.26, 2.58, 2.82, 3.00, 3.12,
        3.08, 2.96, 2.88, 2.96, 3.18, 3.42,
        3.58, 3.64, 3.46, 3.10, 2.64, 2.10,
    ],
    dtype=np.float32,
)
_REAL_PRICE_USD_PER_MWH_HOURLY = np.array(
    [
        68.0, 64.0, 60.0, 58.0, 59.0, 66.0,
        82.0, 98.0, 114.0, 128.0, 136.0, 142.0,
        138.0, 126.0, 118.0, 130.0, 154.0, 178.0,
        196.0, 205.0, 186.0, 152.0, 116.0, 88.0,
    ],
    dtype=np.float32,
)
_REAL_PV_NORM_HOURLY = np.array(
    [
        0.00, 0.00, 0.00, 0.00, 0.00, 0.04,
        0.14, 0.31, 0.52, 0.73, 0.88, 0.96,
        1.00, 0.97, 0.86, 0.69, 0.49, 0.29,
        0.12, 0.03, 0.00, 0.00, 0.00, 0.00,
    ],
    dtype=np.float32,
)
_REAL_WIND_NORM_HOURLY = np.array(
    [
        0.70, 0.67, 0.63, 0.58, 0.55, 0.52,
        0.48, 0.44, 0.41, 0.38, 0.35, 0.33,
        0.32, 0.34, 0.38, 0.44, 0.51, 0.58,
        0.64, 0.69, 0.72, 0.73, 0.72, 0.71,
    ],
    dtype=np.float32,
)


def _upsample_hourly_to_5min(hourly: np.ndarray, horizon: int) -> np.ndarray:
    """Linear interpolation from 24 hourly anchors to horizon 5-minute points."""
    x_hour = np.arange(24, dtype=np.float32)
    x_5min = np.linspace(0.0, 23.0, int(horizon), dtype=np.float32)
    return np.interp(x_5min, x_hour, hourly).astype(np.float32)


def _daily_profiles_5min(horizon: int):
    base_load = _upsample_hourly_to_5min(_LOAD_MW_HOURLY, horizon)
    price_mwh = _upsample_hourly_to_5min(_PRICE_USD_PER_MWH_HOURLY, horizon)
    pv = _upsample_hourly_to_5min(_PV_NORM_HOURLY, horizon)
    wind = _upsample_hourly_to_5min(_WIND_NORM_HOURLY, horizon)

    # Convert USD/MWh to USD per 5-minute interval weighting used by objective.
    # The objective already sums over time slots, so we scale by dt.
    price_slot = price_mwh * FIXED_TIME_STEP_HOURS
    return base_load, price_slot, pv, wind


def _build_real_daily_profiles_5min(horizon: int, country: str, cache_dir: Path, force_download: bool = False):
    # Keep signature for CLI compatibility, but use offline hardcoded anchors only.
    _ = (country, cache_dir, force_download)
    base_5m = _upsample_hourly_to_5min(_REAL_LOAD_MW_HOURLY, horizon)
    price_5m = _upsample_hourly_to_5min(_REAL_PRICE_USD_PER_MWH_HOURLY, horizon) * FIXED_TIME_STEP_HOURS
    pv_5m = _upsample_hourly_to_5min(_REAL_PV_NORM_HOURLY, horizon)
    wind_5m = _upsample_hourly_to_5min(_REAL_WIND_NORM_HOURLY, horizon)
    return base_5m, price_5m, pv_5m, wind_5m


def _build_fixed_multidc_dataset(
    n_samples: int,
    horizon: int,
    num_dc: int,
    num_regions: int,
    num_jobs: int,
    seed: int,
):
    rng = np.random.RandomState(seed)

    base_5m, price_5m, pv_5m, wind_5m = _daily_profiles_5min(horizon)

    # Generate scalable per-DC renewable capacities (MW).
    pv_cap = np.linspace(0.95, 1.35, num_dc, dtype=np.float32)
    wind_cap = np.linspace(0.80, 1.15, num_dc, dtype=np.float32)

    # Region-level interactive load shares.
    region_share = rng.dirichlet(np.ones(num_regions, dtype=np.float32)).astype(np.float32)

    # Batch total work baseline scales with number of jobs.
    batch_work_base = np.linspace(1.4, 3.1, num_jobs, dtype=np.float32)

    base_load = np.zeros((n_samples, horizon), dtype=np.float32)
    price = np.zeros((n_samples, horizon), dtype=np.float32)
    renew_avail = np.zeros((n_samples, num_dc, horizon), dtype=np.float32)
    interactive = np.zeros((n_samples, num_regions, horizon), dtype=np.float32)
    batch_work = np.zeros((n_samples, num_jobs), dtype=np.float32)

    for i in range(n_samples):
        # Small deterministic day-to-day variation.
        amp = 1.0 + 0.04 * rng.randn()
        phase = rng.randint(0, 12)

        base_day = np.roll(base_5m, phase) * amp
        price_day = np.roll(price_5m, phase)

        base_load[i, :] = np.clip(base_day, 1.2, 3.8)
        price[i, :] = np.clip(price_day * (1.0 + 0.03 * rng.randn()), 1.5, 16.0)

        for d in range(num_dc):
            dc_pv = pv_cap[d] * np.roll(pv_5m, d * 5)
            dc_wind = wind_cap[d] * np.roll(wind_5m, d * 7)
            ren = np.clip(dc_pv + 0.55 * dc_wind + 0.03 * rng.randn(horizon), 0.0, None)
            renew_avail[i, d, :] = ren

        inter_total = np.clip(0.34 * base_load[i, :] + 0.03 * rng.randn(horizon), 0.05, None)
        for r in range(num_regions):
            shift = (r - 1) * 3
            interactive[i, r, :] = np.clip(np.roll(inter_total, shift) * region_share[r], 0.01, None)

        bw = batch_work_base * (1.0 + 0.08 * rng.randn(num_jobs))
        batch_work[i, :] = np.clip(bw, 0.6, None)

    data = {
        "base_load": torch.from_numpy(base_load).float(),
        "price": torch.from_numpy(price).float(),
        "renew_avail": torch.from_numpy(renew_avail).float(),
        "interactive_demand": torch.from_numpy(interactive).float(),
        "batch_work": torch.from_numpy(batch_work).float(),
        # flattened mirrors for neural input node
        "renew_avail_flat": torch.from_numpy(renew_avail.reshape(n_samples, -1)).float(),
        "interactive_flat": torch.from_numpy(interactive.reshape(n_samples, -1)).float(),
    }
    return data


def _build_real_multidc_dataset(
    n_samples: int,
    horizon: int,
    num_dc: int,
    num_regions: int,
    num_jobs: int,
    seed: int,
    opsd_country: str,
    cache_dir: Path,
    force_download: bool = False,
):
    rng = np.random.RandomState(seed)
    base_5m, price_5m, pv_5m, wind_5m = _build_real_daily_profiles_5min(
        horizon=horizon,
        country=opsd_country,
        cache_dir=cache_dir,
        force_download=force_download,
    )

    pv_cap = np.linspace(0.95, 1.35, num_dc, dtype=np.float32)
    wind_cap = np.linspace(0.80, 1.15, num_dc, dtype=np.float32)
    region_share = rng.dirichlet(np.ones(num_regions, dtype=np.float32)).astype(np.float32)
    batch_work_base = np.linspace(1.4, 3.1, num_jobs, dtype=np.float32)

    base_load = np.zeros((n_samples, horizon), dtype=np.float32)
    price = np.zeros((n_samples, horizon), dtype=np.float32)
    renew_avail = np.zeros((n_samples, num_dc, horizon), dtype=np.float32)
    interactive = np.zeros((n_samples, num_regions, horizon), dtype=np.float32)
    batch_work = np.zeros((n_samples, num_jobs), dtype=np.float32)

    for i in range(n_samples):
        amp = 1.0 + 0.03 * rng.randn()
        phase = rng.randint(0, 12)

        base_day = np.roll(base_5m, phase) * amp
        price_day = np.roll(price_5m, phase)

        base_load[i, :] = np.clip(base_day, 1.2, 4.0)
        price[i, :] = np.clip(price_day * (1.0 + 0.02 * rng.randn()), 1.2, 20.0)

        for d in range(num_dc):
            dc_pv = pv_cap[d] * np.roll(pv_5m, d * 5)
            dc_wind = wind_cap[d] * np.roll(wind_5m, d * 7)
            ren = np.clip(dc_pv + 0.55 * dc_wind + 0.02 * rng.randn(horizon), 0.0, None)
            renew_avail[i, d, :] = ren

        inter_total = np.clip(0.34 * base_load[i, :] + 0.02 * rng.randn(horizon), 0.05, None)
        for r in range(num_regions):
            shift = (r - 1) * 3
            interactive[i, r, :] = np.clip(np.roll(inter_total, shift) * region_share[r], 0.01, None)

        bw = batch_work_base * (1.0 + 0.06 * rng.randn(num_jobs))
        batch_work[i, :] = np.clip(bw, 0.6, None)

    data = {
        "base_load": torch.from_numpy(base_load).float(),
        "price": torch.from_numpy(price).float(),
        "renew_avail": torch.from_numpy(renew_avail).float(),
        "interactive_demand": torch.from_numpy(interactive).float(),
        "batch_work": torch.from_numpy(batch_work).float(),
        "renew_avail_flat": torch.from_numpy(renew_avail.reshape(n_samples, -1)).float(),
        "interactive_flat": torch.from_numpy(interactive.reshape(n_samples, -1)).float(),
    }
    return data


def _build_dc_bus_map(num_dc: int):
    # Candidate buses on IEEE-33 feeder (0-based, excluding root bus=0).
    candidate = [
        2, 4, 6, 8, 10, 12, 14, 16,
        18, 20, 22, 24, 26, 28, 30, 32,
    ]
    if num_dc > len(candidate):
        raise ValueError(f"num_dc={num_dc} exceeds supported bus-map slots ({len(candidate)}).")
    return candidate[:num_dc]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=17)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--penalty", type=float, default=40.0)
    parser.add_argument("--eq_weight", type=float, default=2.0)
    parser.add_argument("--obj_weight", type=float, default=1.0)
    parser.add_argument("--viol_weight", type=float, default=1.0)

    parser.add_argument("--hlayers_sol", type=int, default=6)
    parser.add_argument("--hlayers_rnd", type=int, default=4)
    parser.add_argument("--hsize", type=int, default=128)
    parser.add_argument("--smap_arch", type=str, default="mlp", choices=["mlp", "lstm", "rnn", "tcn"])
    parser.add_argument("--smap_head_mode", type=str, default="single", choices=["single", "dual"])
    parser.add_argument("--smap_residual", type=int, default=0)
    parser.add_argument("--smap_residual_scale", type=float, default=0.3)
    parser.add_argument("--constraint_feat_inject", type=int, default=0)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.1)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--tcn_blocks", type=int, default=4)
    parser.add_argument("--tcn_dropout", type=float, default=0.1)

    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--train_eval_batches", type=int, default=8)

    parser.add_argument("--penalty_growth", action="store_true")
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--tb_logdir", type=str, default="runs")

    parser.add_argument("--compare_solver", action="store_true")
    parser.add_argument("--solver_time_limit", type=float, default=30.0)
    parser.add_argument("--solver_tee", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=50)

    parser.add_argument("--method", type=str, default="cls", choices=["cls", "solver"])

    parser.add_argument("--horizon", type=int, default=FIXED_HORIZON)
    parser.add_argument("--num_dc", type=int, default=FIXED_NUM_DC)
    parser.add_argument("--num_regions", type=int, default=FIXED_NUM_REGIONS)
    parser.add_argument("--num_jobs", type=int, default=FIXED_NUM_JOBS)
    parser.add_argument("--solver", type=str, default=FIXED_SOLVER)

    parser.add_argument("--data_profile", type=str, default="fixed", choices=["fixed", "real"])
    parser.add_argument("--opsd_country", type=str, default="DE")
    parser.add_argument("--real_data_force_download", action="store_true")
    parser.add_argument("--real_data_fallback_to_fixed", action="store_true")

    parser.add_argument("--dc_nodes_per_dc", type=int, default=3)
    parser.add_argument("--cpu_per_node", type=float, default=1.0)

    args = parser.parse_args()

    if args.horizon <= 0 or args.num_dc <= 1 or args.num_regions <= 0 or args.num_jobs <= 0:
        raise ValueError("Require horizon>0, num_dc>1, num_regions>0, num_jobs>0")
    if args.dc_nodes_per_dc <= 0 or args.cpu_per_node <= 0.0:
        raise ValueError("Require dc_nodes_per_dc>0 and cpu_per_node>0")

    # Expose scalable per-DC node/capacity settings to downstream model builders.
    cpu_per_node_scalar = float(args.cpu_per_node)
    args.num_nodes_per_dc = [int(args.dc_nodes_per_dc)] * int(args.num_dc)
    args.cpu_per_node = [cpu_per_node_scalar] * int(args.num_dc)
    args.dc_cpu_cap = [float(args.dc_nodes_per_dc) * cpu_per_node_scalar] * int(args.num_dc)
    args.dc_bus_map = _build_dc_bus_map(int(args.num_dc))

    print(
        "[Multi-DC setup] "
        f"horizon={args.horizon} (5min/step), "
        f"num_dc={args.num_dc}, num_regions={args.num_regions}, num_jobs={args.num_jobs}, "
        f"dc_nodes_per_dc={args.dc_nodes_per_dc}, dc_cpu_cap={args.dc_cpu_cap[0]:.2f}, solver={args.solver}"
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    n_total = args.train_size + args.val_size + args.test_size
    if str(args.data_profile).lower() == "real":
        try:
            cache_dir = Path(__file__).resolve().parent / "data_cache"
            data = _build_real_multidc_dataset(
                n_samples=n_total,
                horizon=args.horizon,
                num_dc=args.num_dc,
                num_regions=args.num_regions,
                num_jobs=args.num_jobs,
                seed=args.data_seed,
                opsd_country=str(args.opsd_country).upper(),
                cache_dir=cache_dir,
                force_download=bool(args.real_data_force_download),
            )
            print("[Data] Using hardcoded real-like profile (offline, no download).")
        except Exception as e:
            if bool(args.real_data_fallback_to_fixed):
                print(f"[Data][Warning] Failed to build real data profile: {e}")
                print("[Data][Warning] Falling back to fixed synthetic profile.")
                data = _build_fixed_multidc_dataset(
                    n_samples=n_total,
                    horizon=args.horizon,
                    num_dc=args.num_dc,
                    num_regions=args.num_regions,
                    num_jobs=args.num_jobs,
                    seed=args.data_seed,
                )
            else:
                raise
    else:
        data = _build_fixed_multidc_dataset(
            n_samples=n_total,
            horizon=args.horizon,
            num_dc=args.num_dc,
            num_regions=args.num_regions,
            num_jobs=args.num_jobs,
            seed=args.data_seed,
        )

    train_ds, test_ds, val_ds = data_split(
        data,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.split_seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, collate_fn=train_ds.collate_fn, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0, collate_fn=test_ds.collate_fn, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, collate_fn=val_ds.collate_fn, shuffle=False)

    if args.method == "solver":
        run.multi_dc.exact(test_loader, args)
    else:
        run.multi_dc.rndCls(train_loader, test_loader, val_loader, args, penalty_growth=args.penalty_growth)


if __name__ == "__main__":
    main()

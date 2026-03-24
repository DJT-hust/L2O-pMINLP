#!/usr/bin/env python
# coding: utf-8
"""
Entry script for Microgrid Scheduling experiments (pMINLP).
Uses separate input keys: load/pv/price_buy/price_sell/soc0.
Default horizon T=24.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import run


def generate_microgrid_dataset(num_samples: int, T: int, seed: int = 17):
    """
    Generate synthetic microgrid scenarios.
    Returns dict of tensors (float32):
      load: [N,T]
      pv: [N,T]
      price_buy: [N,T]
      price_sell: [N,T]
      soc0: [N,1]
    """
    rng = np.random.RandomState(seed)

    # load: base + daily pattern + noise, clipped
    base_load = 1.3 + 0.2 * rng.rand(num_samples, 1)
    daily = (np.sin(np.linspace(0, 2 * np.pi, T, endpoint=False)) + 1.0) / 2.0  # [0,1]
    daily = daily.reshape(1, T)
    load = base_load + 0.6 * daily + 0.1 * rng.randn(num_samples, T)
    load = np.clip(load, 0.2, 3.0)

    # pv: daytime bell-ish curve + noise, clipped
    sun = np.maximum(0.0, np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, T, endpoint=False)))
    sun = sun.reshape(1, T)
    pv = 0.8 * sun + 0.05 * rng.randn(num_samples, T)
    pv = np.clip(pv, 0.0, 1.2)

    # price_buy: TOU-ish, higher at peaks
    price_buy = 0.4 + 0.2 * daily + 0.05 * rng.rand(num_samples, T)
    price_buy = np.clip(price_buy, 0.05, 1.0)

    # price_sell: lower than buy
    price_sell = 0.2 + 0.05 * daily + 0.03 * rng.rand(num_samples, T)
    price_sell = np.clip(price_sell, 0.01, 0.6)

    # soc0 in [0.1, 4.0] (match math solver defaults)
    soc0 = 0.1 + (4.0 - 0.1) * rng.rand(num_samples, 1)

    data = {
        "load": torch.from_numpy(load).float(),
        "pv": torch.from_numpy(pv).float(),
        "price_buy": torch.from_numpy(price_buy).float(),
        "price_sell": torch.from_numpy(price_sell).float(),
        "soc0": torch.from_numpy(soc0).float(),
    }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train_size", type=int, default=8000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--penalty", type=float, default=50.0)

    parser.add_argument("--hlayers_sol", type=int, default=5)
    parser.add_argument("--hlayers_rnd", type=int, default=4)
    parser.add_argument("--hsize", type=int, default=64)

    parser.add_argument("--project", action="store_true")
    parser.add_argument("--penalty_growth", action="store_true")
    parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging for training")
    parser.add_argument("--tb_logdir", type=str, default="runs", help="TensorBoard log root directory")

    # projection hyperparams
    parser.add_argument("--proj_iters", type=int, default=200)
    parser.add_argument("--proj_step", type=float, default=1e-2)
    parser.add_argument("--proj_decay", type=float, default=1.0)

    # which method to run
    parser.add_argument("--method", type=str, default="cls", choices=["cls", "thd", "ste"])
    args = parser.parse_args()

    # build dataset
    N = args.train_size + args.val_size + args.test_size
    data = generate_microgrid_dataset(N, args.horizon, seed=17)

    # split
    idx = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)

    train_idx = idx[: args.train_size]
    val_idx = idx[args.train_size : args.train_size + args.val_size]
    test_idx = idx[args.train_size + args.val_size :]

    def subset(d, inds):
        return {k: v[inds] for k, v in d.items()}

    from src.utlis import DictDataset  # assuming repo has this; if not, tell me and I'll adapt
    train_ds = DictDataset(subset(data, train_idx))
    val_ds = DictDataset(subset(data, val_idx))
    test_ds = DictDataset(subset(data, test_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # run
    import run.microgrid as mg
    if args.method == "cls":
        mg.rndCls(train_loader, test_loader, val_loader, args, penalty_growth=args.penalty_growth)
    elif args.method == "thd":
        mg.rndThd(train_loader, test_loader, val_loader, args, penalty_growth=args.penalty_growth)
    else:
        mg.rndSte(train_loader, test_loader, val_loader, args, penalty_growth=args.penalty_growth)


if __name__ == "__main__":
    main()
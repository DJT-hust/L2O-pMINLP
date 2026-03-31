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
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for model/training randomness")
    parser.add_argument("--data_seed", type=int, default=17,
                        help="Seed used to generate synthetic microgrid dataset")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed used to split train/val/test indices")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train_size", type=int, default=8000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_anneal", action="store_true",
                        help="Enable cosine annealing schedule for learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum learning rate when --lr_anneal is enabled")
    parser.add_argument("--penalty", type=float, default=50.0)
    parser.add_argument("--obj_weight", type=float, default=1.0,
                        help="Weight for objective term in training loss")
    parser.add_argument("--viol_weight", type=float, default=1.0,
                        help="Weight for violation term in training loss")
    parser.add_argument("--viol_threshold", type=float, default=0.0,
                        help="Hinge threshold for violation term in training loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience measured on validation checks")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Validation-check warmup before early stopping; default is 20 (or 50 with --penalty_growth)")
    parser.add_argument("--validate_every", type=int, default=125,
                        help="Run validation/logging every N training iterations")
    parser.add_argument("--train_eval_batches", type=int, default=8,
                        help="Number of training batches used in eval-mode train loss; 0 means full train loader")
    parser.add_argument("--loss_report_offset", type=float, default=0.0,
                        help="Add this constant to real train/eval loss values (affects optimization and logged losses)")

    parser.add_argument("--hlayers_sol", type=int, default=8)
    parser.add_argument("--hlayers_rnd", type=int, default=6)
    parser.add_argument("--hsize", type=int, default=128)
    parser.add_argument("--smap_arch", type=str, default="hybrid", choices=["hybrid", "dual_tcn"],
                        help="Solution-map architecture: hybrid (TCN+Transformer) or dual_tcn")
    parser.add_argument("--temporal_blocks", type=int, default=4,
                        help="Number of temporal residual blocks in the dual-head policy")
    parser.add_argument("--temporal_dropout", type=float, default=0.1,
                        help="Dropout used in temporal dual-head policy")
    parser.add_argument("--residual_scale", type=float, default=0.6,
                        help="Scale factor for residual policy delta added to heuristic baseline")
    parser.add_argument("--refine_scale", type=float, default=0.15,
                        help="Scale factor for cross-variable post-refiner residual in policy output")
    parser.add_argument("--transformer_layers", type=int, default=2,
                        help="Number of transformer encoder layers for hybrid smap")
    parser.add_argument("--transformer_heads", type=int, default=4,
                        help="Number of attention heads for hybrid smap")

    parser.add_argument("--project", action="store_true")
    parser.add_argument("--penalty_growth", action="store_true")
    parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging for training")
    parser.add_argument("--tb_logdir", type=str, default="runs", help="TensorBoard log root directory")
    parser.add_argument("--compare_solver", action="store_true",
                        help="Evaluate solver baseline on the same test cases")
    parser.add_argument("--policy_select", type=str, default="best", choices=["best", "proj", "no_proj"],
                        help="How to select final policy candidate at evaluation time")
    parser.add_argument("--policy_feas_guard", type=float, default=1e-4,
                        help="Mean violation threshold used by best-selector to prefer feasible-safe candidates")
    parser.add_argument("--policy_mc_samples", type=int, default=1,
                        help="Number of no-proj policy candidates evaluated per sample (>=1)")
    parser.add_argument("--policy_mc_noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to x_rnd for MC candidate generation")
    parser.add_argument("--policy_mc_seed", type=int, default=123,
                        help="Random seed for MC candidate generation")
    parser.add_argument("--solver_time_limit", type=float, default=60.0,
                        help="Per-case solver time limit in seconds for baseline evaluation")
    parser.add_argument("--solver_tee", action="store_true",
                        help="Print solver logs during baseline evaluation")
    parser.add_argument("--eval_all_test", action="store_true",
                        help="Evaluate on the full test split instead of the default first 100 samples")

    # optional solver-guided distillation on a subset of training data
    parser.add_argument("--distill_ratio", type=float, default=0.0,
                        help="Fraction of training samples solved by solver for distillation labels")
    parser.add_argument("--distill_weight", type=float, default=0.0,
                        help="Weight of solver-guided distillation loss term")
    parser.add_argument("--distill_time_limit", type=float, default=10.0,
                        help="Per-sample solver time limit (seconds) for distillation label generation")
    parser.add_argument("--distill_seed", type=int, default=123,
                        help="Random seed for selecting distillation subset")
    parser.add_argument("--distill_solver_tee", action="store_true",
                        help="Print solver logs while generating distillation labels")

    # projection hyperparams
    parser.add_argument("--proj_iters", type=int, default=200)
    parser.add_argument("--proj_step", type=float, default=1e-2)
    parser.add_argument("--proj_decay", type=float, default=1.0)
    parser.add_argument("--proj_obj_guard", type=float, default=0.0,
                        help="Objective guard weight in projection metric: viol + w*obj")
    parser.add_argument("--proj_trigger_mean_viol", type=float, default=1e-3,
                        help="Run projection only if no-projection mean violation exceeds this threshold")
    parser.add_argument("--proj_viol_tol", type=float, default=1e-6,
                        help="Projection stopping tolerance on max violation")
    parser.add_argument("--proj_early_stop_patience", type=int, default=20,
                        help="Stop projection when violation improvement stalls for this many iterations")
    parser.add_argument("--proj_min_viol_improve", type=float, default=1e-8,
                        help="Minimum mean-violation improvement to reset projection stall counter")
    parser.add_argument("--proj_restarts", type=int, default=1,
                        help="Number of random restarts for projection")
    parser.add_argument("--proj_restart_noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to x at each projection restart (except first)")
    parser.add_argument("--proj_tail_steps", type=int, default=0,
                        help="Number of last timesteps to upweight in projection objective guard")
    parser.add_argument("--proj_tail_weight", type=float, default=1.0,
                        help="Weight multiplier on last proj_tail_steps timesteps in projection objective guard")
    parser.add_argument("--proj_end_discharge_steps", type=int, default=0,
                        help="Apply tighter discharge cap for last K timesteps in hard repair")
    parser.add_argument("--proj_end_discharge_scale", type=float, default=1.0,
                        help="Scale (<1 tightens) for discharge cap on last proj_end_discharge_steps timesteps")

    # which method to run
    parser.add_argument("--method", type=str, default="cls", choices=["cls", "thd", "ste"])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # build dataset
    N = args.train_size + args.val_size + args.test_size
    data = generate_microgrid_dataset(N, args.horizon, seed=args.data_seed)

    # split
    idx = np.arange(N)
    rng = np.random.RandomState(args.split_seed)
    rng.shuffle(idx)

    train_idx = idx[: args.train_size]
    val_idx = idx[args.train_size : args.train_size + args.val_size]
    test_idx = idx[args.train_size + args.val_size :]

    def subset(d, inds):
        return {k: v[inds] for k, v in d.items()}

    def build_distill_labels(train_data, horizon, ratio, timelimit, seed, tee=False):
        from src.problem.math_solver.microgrid import microgrid as msMicrogrid

        n = train_data["load"].shape[0]
        x_teacher = torch.zeros((n, 9 * horizon), dtype=torch.float32)
        teacher_mask = torch.zeros((n, 1), dtype=torch.float32)

        if ratio <= 0.0:
            train_data["x_teacher"] = x_teacher
            train_data["teacher_mask"] = teacher_mask
            return

        m = msMicrogrid(horizon=horizon, timelimit=timelimit)
        rng = np.random.RandomState(seed)
        k = int(max(1, min(n, round(ratio * n))))
        sel = rng.choice(n, size=k, replace=False)

        var_order = ["p_grid_buy", "p_grid_sell", "p_gen", "p_ch", "p_dis", "s_load", "u_gen", "u_ch", "u_dis"]
        ok = 0
        for idx in sel:
            params = {
                "load": train_data["load"][idx].detach().cpu().numpy(),
                "pv": train_data["pv"][idx].detach().cpu().numpy(),
                "price_buy": train_data["price_buy"][idx].detach().cpu().numpy(),
                "price_sell": train_data["price_sell"][idx].detach().cpu().numpy(),
                "soc0": float(train_data["soc0"][idx].detach().cpu().numpy().reshape(-1)[0]),
            }
            try:
                m.set_param_val(params)
                solvals, _ = m.solve(tee=tee)
                if solvals is None:
                    continue

                x = np.zeros(9 * horizon, dtype=np.float32)
                for vname in var_order:
                    sl = m.x_slices[vname]
                    for t in range(horizon):
                        x[sl.start + t] = float(solvals[vname][t])

                x_teacher[idx] = torch.from_numpy(x)
                teacher_mask[idx, 0] = 1.0
                ok += 1
            except Exception:
                continue

        print(f"[Distill] Labeled samples: {ok}/{k} selected ({ok/max(1,n):.2%} of train)")
        train_data["x_teacher"] = x_teacher
        train_data["teacher_mask"] = teacher_mask

    from src.utlis import DictDataset  # assuming repo has this; if not, tell me and I'll adapt
    train_data = subset(data, train_idx)
    build_distill_labels(
        train_data,
        horizon=args.horizon,
        ratio=args.distill_ratio,
        timelimit=args.distill_time_limit,
        seed=args.distill_seed,
        tee=args.distill_solver_tee,
    )
    train_ds = DictDataset(train_data)
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
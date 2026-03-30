#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for Microgrid Scheduling (MIQP) using pMINLP.
Inputs are provided as separate keys: load/pv/price_buy/price_sell/soc0.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from run import utils


def _to_device(batch, device="cuda"):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def _concat_params(datadict, keys):
    """Concatenate params into xi for networks that want a single tensor."""
    return torch.cat([datadict[k] for k in keys], dim=-1)


def _microgrid_constraint_labels(horizon):
    """Build readable labels for constraints in the exact add-order of math_solver.microgrid."""
    labels = []
    for t in range(horizon):
        labels.extend([
            f"p_grid_def[t={t}]",        # p_grid = p_grid_buy - p_grid_sell
            f"power_balance[t={t}]",     # power balance equality
            f"gen_cap[t={t}]",           # p_gen <= p_gen_max * u_gen
            f"ch_cap[t={t}]",            # p_ch <= p_ch_max * u_ch
            f"dis_cap[t={t}]",           # p_dis <= p_dis_max * u_dis
            f"ch_dis_mutex[t={t}]",      # u_ch + u_dis <= 1
            f"soc_lb[t={t}]",            # soc >= soc_min
            f"soc_ub[t={t}]",            # soc <= soc_max
            f"soc_dyn[t={t}]",           # SOC dynamics
        ])
    return labels


def _reconstruct_soc_from_x_numpy(x, x_slices, horizon, soc0, eta_ch, eta_dis):
    """
    Reconstruct SOC values from charge/discharge decisions.
    NOTE: SOC is no longer part of the flattened x (9T instead of 10T).
    This function computes SOC directly and returns it separately.
    """
    ch_sl = x_slices["p_ch"]
    dis_sl = x_slices["p_dis"]

    p_ch = x[ch_sl]
    p_dis = x[dis_sl]
    soc = np.zeros(horizon, dtype=float)

    soc[0] = float(soc0 + eta_ch * p_ch[0] - (1.0 / eta_dis) * p_dis[0])
    for t in range(1, horizon):
        soc[t] = float(soc[t - 1] + eta_ch * p_ch[t] - (1.0 / eta_dis) * p_dis[t])

    return soc  # Return SOC directly (no longer write back into x)


def _enforce_power_balance_in_x_numpy(
    x,
    x_slices,
    load,
    pv,
    p_grid_buy_max=None,
    p_grid_sell_max=None,
):
    """
    Enforce power balance with an economic preference:
    1) satisfy mismatch via grid buy/sell first,
    2) use s_load as fallback only when grid caps are tight.
    """
    buy_sl = x_slices["p_grid_buy"]
    sell_sl = x_slices["p_grid_sell"]
    gen_sl = x_slices["p_gen"]
    ch_sl = x_slices["p_ch"]
    dis_sl = x_slices["p_dis"]
    s_load_sl = x_slices["s_load"]

    p_gen = x[gen_sl]
    p_ch = x[ch_sl]
    p_dis = x[dis_sl]
    s_load = x[s_load_sl]

    # Keep shed within physical bounds before balancing.
    s_load = np.clip(s_load, 0.0, load)

    # Residual grid injection needed by power balance.
    # required_grid > 0: need import (buy), < 0: need export (sell).
    required_grid = load - p_gen - p_dis + p_ch - pv - s_load

    buy_cap = np.inf if p_grid_buy_max is None else float(p_grid_buy_max)
    sell_cap = np.inf if p_grid_sell_max is None else float(p_grid_sell_max)
    p_grid_buy = np.clip(required_grid, 0.0, buy_cap)
    p_grid_sell = np.clip(-required_grid, 0.0, sell_cap)

    # If caps prevent exact balance, absorb the remaining mismatch in s_load.
    p_grid = p_grid_buy - p_grid_sell
    s_load = load - p_gen - p_dis + p_ch - p_grid - pv
    s_load = np.clip(s_load, 0.0, load)

    x[buy_sl] = p_grid_buy
    x[sell_sl] = p_grid_sell
    x[s_load_sl] = s_load
    return x


def _fast_hard_repair_x_numpy(x, model, soc0=None):
    """
    Fast deterministic repair (no gradients):
    - clip binary controls to {0,1}
    - enforce charge/discharge mutex
    - enforce nonnegativity and power-cap constraints
    This is intentionally lightweight and used for no-projection evaluation mode.
    """
    x = np.asarray(x, dtype=float).copy()
    s = model.x_slices
    t = model.horizon

    sl_buy = s["p_grid_buy"]
    sl_sell = s["p_grid_sell"]
    sl_gen = s["p_gen"]
    sl_ch = s["p_ch"]
    sl_dis = s["p_dis"]
    sl_u_gen = s["u_gen"]
    sl_u_ch = s["u_ch"]
    sl_u_dis = s["u_dis"]

    p_grid_buy = x[sl_buy]
    p_grid_sell = x[sl_sell]
    p_gen = x[sl_gen]
    p_ch = x[sl_ch]
    p_dis = x[sl_dis]
    u_gen = (x[sl_u_gen] >= 0.5).astype(float)
    u_ch = (x[sl_u_ch] >= 0.5).astype(float)
    u_dis = (x[sl_u_dis] >= 0.5).astype(float)

    # Strict mutex for battery mode.
    dual_on = (u_ch + u_dis) > 1.0
    keep_ch = u_ch >= u_dis
    u_ch = np.where(dual_on & keep_ch, 1.0, u_ch)
    u_dis = np.where(dual_on & keep_ch, 0.0, u_dis)
    u_ch = np.where(dual_on & (~keep_ch), 0.0, u_ch)
    u_dis = np.where(dual_on & (~keep_ch), 1.0, u_dis)

    # Capacity + nonnegativity repair.
    p_grid_buy = np.clip(p_grid_buy, 0.0, model.p_grid_buy_max)
    p_grid_sell = np.clip(p_grid_sell, 0.0, model.p_grid_sell_max)
    p_gen = np.clip(p_gen, 0.0, model.p_gen_max * u_gen)
    p_ch = np.clip(p_ch, 0.0, model.p_ch_max * u_ch)
    p_dis = np.clip(p_dis, 0.0, model.p_dis_max * u_dis)

    # Conservative tail discharge tightening to reduce end-horizon capacity spikes.
    tail_k = int(max(0, getattr(model, "horizon", t)))
    if tail_k > 0:
        k = min(2, t)
        if k > 0:
            p_dis[-k:] = np.minimum(p_dis[-k:], 0.95 * model.p_dis_max * u_dis[-k:])

    # SOC-feasibility repair (forward pass): clip p_dis/p_ch by reachable SOC bounds.
    if soc0 is not None:
        soc = float(soc0)
        for i in range(t):
            max_dis_by_soc = max(0.0, (soc - model.soc_min) * model.eta_dis)
            p_dis[i] = min(p_dis[i], max_dis_by_soc)

            max_ch_by_soc = max(0.0, (model.soc_max - soc) / max(model.eta_ch, 1e-12))
            p_ch[i] = min(p_ch[i], max_ch_by_soc)

            soc = soc + model.eta_ch * p_ch[i] - (1.0 / model.eta_dis) * p_dis[i]

    x[sl_buy] = p_grid_buy
    x[sl_sell] = p_grid_sell
    x[sl_gen] = p_gen
    x[sl_ch] = p_ch
    x[sl_dis] = p_dis
    x[sl_u_gen] = u_gen
    x[sl_u_ch] = u_ch
    x[sl_u_dis] = u_dis
    return x


def _economic_post_balance_repair_x_numpy(x, model):
    """
    Economic repair on already-balanced x:
    - reduce load shedding first by increasing generator output (within cap),
    - then increase grid buy (within cap),
    while preserving power balance identity at each step.
    """
    x = np.asarray(x, dtype=float).copy()
    s = model.x_slices
    t = model.horizon

    sl_buy = s["p_grid_buy"]
    sl_sell = s["p_grid_sell"]
    sl_gen = s["p_gen"]
    sl_shed = s["s_load"]
    sl_u_gen = s["u_gen"]

    p_grid_buy = x[sl_buy]
    p_grid_sell = x[sl_sell]
    p_gen = x[sl_gen]
    s_load = x[sl_shed]
    u_gen = x[sl_u_gen]

    # Enforce one-sided grid exchange by net decomposition.
    p_grid_net = p_grid_buy - p_grid_sell
    p_grid_buy = np.clip(p_grid_net, 0.0, model.p_grid_buy_max)
    p_grid_sell = np.clip(-p_grid_net, 0.0, model.p_grid_sell_max)

    deficit = np.maximum(s_load, 0.0)

    # 1) Use generator headroom first (usually cheaper than load shedding penalty).
    gen_room = np.maximum(float(model.p_gen_max) - p_gen, 0.0)
    dg = np.minimum(deficit, gen_room)
    p_gen = p_gen + dg
    deficit = deficit - dg
    u_gen = np.where(dg > 1e-12, 1.0, u_gen)

    # 2) Use grid import headroom next.
    buy_room = np.maximum(float(model.p_grid_buy_max) - p_grid_buy, 0.0)
    db = np.minimum(deficit, buy_room)
    p_grid_buy = p_grid_buy + db
    deficit = deficit - db

    s_load = deficit

    x[sl_buy] = p_grid_buy
    x[sl_sell] = p_grid_sell
    x[sl_gen] = p_gen
    x[sl_shed] = s_load
    x[sl_u_gen] = np.clip(u_gen, 0.0, 1.0)
    return x


def _build_solution_map_policy(config, model, horizon):
    from src.func.layer import DualHeadTemporalResidualPolicy, DualHeadHybridTemporalPolicy

    arch = getattr(config, "smap_arch", "hybrid")
    if arch == "dual_tcn":
        return DualHeadTemporalResidualPolicy(
            horizon=horizon,
            out_dim=model.nx,
            hidden_dim=config.hsize,
            num_blocks=getattr(config, "temporal_blocks", 4),
            dropout=getattr(config, "temporal_dropout", 0.1),
            residual_scale=getattr(config, "residual_scale", 0.6),
            refine_scale=getattr(config, "refine_scale", 0.15),
        )

    return DualHeadHybridTemporalPolicy(
        horizon=horizon,
        out_dim=model.nx,
        hidden_dim=config.hsize,
        num_blocks=getattr(config, "temporal_blocks", 4),
        dropout=getattr(config, "temporal_dropout", 0.1),
        residual_scale=getattr(config, "residual_scale", 0.6),
        refine_scale=getattr(config, "refine_scale", 0.15),
        tf_layers=getattr(config, "transformer_layers", 2),
        tf_heads=getattr(config, "transformer_heads", 4),
    )


def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    """
    Learned rounding via Gumbel (Classifier-style).
    """
    print(config)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC in Microgrid for horizon {config.horizon}.")

    import neuromancer as nm
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    from src.problem.math_solver.microgrid import microgrid as msMicrogrid
    from src.problem.neuromancer.microgrid import penaltyLoss as nmMicrogridLoss

    T = config.horizon
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project

    # math model (for int/bin indices + evaluation)
    model = msMicrogrid(horizon=T, timelimit=1000)

    # parameter dimension (separate keys)
    param_keys = ["load", "pv", "price_buy", "price_sell", "soc0"]
    param_dim = 4 * T + 1  # load(T) + pv(T) + buy(T) + sell(T) + soc0(1)

    # solution map: temporal dual-head residual policy
    func = _build_solution_map_policy(config, model, T)

    class ParamConcatNode(nn.Module):
        def forward(self, *inputs):
            xi = torch.cat(list(inputs), dim=-1)
            return xi

    concat_node = nm.system.Node(ParamConcatNode(), param_keys, ["xi"], name="concat")
    smap = nm.system.Node(func, ["xi"], ["x"], name="smap")

    # rounding network + rounding model
    layers_rnd = netFC(input_dim=param_dim + model.nx, hidden_dims=[hsize] * hlayers_rnd, output_dim=model.nx)

    rnd = roundGumbelModel(
        layers=layers_rnd,
        param_keys=["xi"],
        var_keys=["x"],
        output_keys=["x_rnd"],
        int_ind=model.int_ind,
        bin_ind=model.bin_ind,
        continuous_update=True,
        name="round",
    )

    components = nn.ModuleList([concat_node, smap, rnd]).to("cuda")

    loss_fn = nmMicrogridLoss(
        ["load", "pv", "price_buy", "price_sell", "soc0", "x_rnd"],
        horizon=T,
        penalty_weight=penalty_weight,
        # keep consistent with math_solver defaults unless you override
        p_gen_max=model.p_gen_max,
        p_ch_max=model.p_ch_max,
        p_dis_max=model.p_dis_max,
        soc_min=model.soc_min,
        soc_max=model.soc_max,
        eta_ch=model.eta_ch,
        eta_dis=model.eta_dis,
        gen_quad=model.gen_quad,
        gen_lin=model.gen_lin,
        gen_on_cost=model.gen_on_cost,
        load_shed_penalty=model.load_shed_penalty,
        eq_weight=5.0,  # increase power balance constraint weight for better constraint satisfaction
        obj_weight=getattr(config, "obj_weight", 1.0),
        viol_weight=getattr(config, "viol_weight", 1.0),
        viol_threshold=getattr(config, "viol_threshold", 0.0),
        distill_weight=getattr(config, "distill_weight", 0.0),
    )

    utils.train(
        components,
        loss_fn,
        loader_train,
        loader_val,
        lr,
        penalty_growth,
        patience=getattr(config, "patience", 20),
        warmup=getattr(config, "warmup", None),
        validate_every=getattr(config, "validate_every", 125),
        train_eval_batches=getattr(config, "train_eval_batches", 8),
        lr_anneal=getattr(config, "lr_anneal", False),
        lr_min=getattr(config, "lr_min", 1e-6),
        loader_test=loader_test,
        tensorboard=getattr(config, "tb", False),
        tb_logdir=getattr(config, "tb_logdir", "runs"),
        tb_run_name=f"microgrid_cls_T{T}_pen{penalty_weight}",
    )
    df = evaluate(components, loss_fn, model, loader_test, project, config)

    tag = "cls"
    suffix = "-g" if penalty_growth else ("-p" if project else "")
    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_dir / f"mg_{tag}{penalty_weight}_T{T}{suffix}.csv")


def rndThd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    """
    Learned rounding via learned thresholds.
    """
    print(config)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT in Microgrid for horizon {config.horizon}.")

    import neuromancer as nm
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    from src.problem.math_solver.microgrid import microgrid as msMicrogrid
    from src.problem.neuromancer.microgrid import penaltyLoss as nmMicrogridLoss

    T = config.horizon
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project

    model = msMicrogrid(horizon=T, timelimit=1000)

    param_keys = ["load", "pv", "price_buy", "price_sell", "soc0"]
    param_dim = 4 * T + 1

    func = _build_solution_map_policy(config, model, T)

    class ParamConcatNode(nn.Module):
        def forward(self, *inputs):
            xi = torch.cat(list(inputs), dim=-1)
            return xi

    concat_node = nm.system.Node(ParamConcatNode(), param_keys, ["xi"], name="concat")
    smap = nm.system.Node(func, ["xi"], ["x"], name="smap")

    layers_rnd = netFC(input_dim=param_dim + model.nx, hidden_dims=[hsize] * hlayers_rnd, output_dim=model.nx)

    rnd = roundThresholdModel(
        layers=layers_rnd,
        param_keys=["xi"],
        var_keys=["x"],
        output_keys=["x_rnd"],
        int_ind=model.int_ind,
        bin_ind=model.bin_ind,
        continuous_update=True,
        name="round",
    )

    components = nn.ModuleList([concat_node, smap, rnd]).to("cuda")

    loss_fn = nmMicrogridLoss(
        ["load", "pv", "price_buy", "price_sell", "soc0", "x_rnd"],
        horizon=T,
        penalty_weight=penalty_weight,
        p_gen_max=model.p_gen_max,
        p_ch_max=model.p_ch_max,
        p_dis_max=model.p_dis_max,
        soc_min=model.soc_min,
        soc_max=model.soc_max,
        eta_ch=model.eta_ch,
        eta_dis=model.eta_dis,
        gen_quad=model.gen_quad,
        gen_lin=model.gen_lin,
        gen_on_cost=model.gen_on_cost,
        load_shed_penalty=model.load_shed_penalty,
        eq_weight=5.0,  # increase power balance constraint weight for better constraint satisfaction
        obj_weight=getattr(config, "obj_weight", 1.0),
        viol_weight=getattr(config, "viol_weight", 1.0),
        viol_threshold=getattr(config, "viol_threshold", 0.0),
        distill_weight=getattr(config, "distill_weight", 0.0),
    )

    utils.train(
        components,
        loss_fn,
        loader_train,
        loader_val,
        lr,
        penalty_growth,
        patience=getattr(config, "patience", 20),
        warmup=getattr(config, "warmup", None),
        validate_every=getattr(config, "validate_every", 125),
        train_eval_batches=getattr(config, "train_eval_batches", 8),
        lr_anneal=getattr(config, "lr_anneal", False),
        lr_min=getattr(config, "lr_min", 1e-6),
        loader_test=loader_test,
        tensorboard=getattr(config, "tb", False),
        tb_logdir=getattr(config, "tb_logdir", "runs"),
        tb_run_name=f"microgrid_thd_T{T}_pen{penalty_weight}",
    )
    df = evaluate(components, loss_fn, model, loader_test, project, config)

    tag = "thd"
    suffix = "-g" if penalty_growth else ("-p" if project else "")
    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_dir / f"mg_{tag}{penalty_weight}_T{T}{suffix}.csv")


def rndSte(loader_train, loader_test, loader_val, config, penalty_growth=False):
    """
    STE rounding (no learned rounding network).
    """
    print(config)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RS in Microgrid for horizon {config.horizon}.")

    import neuromancer as nm
    from src.func import roundSTEModel
    from src.problem.math_solver.microgrid import microgrid as msMicrogrid
    from src.problem.neuromancer.microgrid import penaltyLoss as nmMicrogridLoss

    T = config.horizon
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project

    model = msMicrogrid(horizon=T, timelimit=1000)

    param_keys = ["load", "pv", "price_buy", "price_sell", "soc0"]
    param_dim = 4 * T + 1

    func = _build_solution_map_policy(config, model, T)

    class ParamConcatNode(nn.Module):
        def forward(self, *inputs):
            xi = torch.cat(list(inputs), dim=-1)
            return xi

    concat_node = nm.system.Node(ParamConcatNode(), param_keys, ["xi"], name="concat")
    smap = nm.system.Node(func, ["xi"], ["x"], name="smap")

    rnd = roundSTEModel(
        param_keys=["xi"],
        var_keys=["x"],
        output_keys=["x_rnd"],
        int_ind=model.int_ind,
        bin_ind=model.bin_ind,
        name="round",
    )

    components = nn.ModuleList([concat_node, smap, rnd]).to("cuda")

    loss_fn = nmMicrogridLoss(
        ["load", "pv", "price_buy", "price_sell", "soc0", "x_rnd"],
        horizon=T,
        penalty_weight=penalty_weight,
        p_gen_max=model.p_gen_max,
        p_ch_max=model.p_ch_max,
        p_dis_max=model.p_dis_max,
        soc_min=model.soc_min,
        soc_max=model.soc_max,
        eta_ch=model.eta_ch,
        eta_dis=model.eta_dis,
        gen_quad=model.gen_quad,
        gen_lin=model.gen_lin,
        gen_on_cost=model.gen_on_cost,
        load_shed_penalty=model.load_shed_penalty,
        eq_weight=5.0,  # increase power balance constraint weight for better constraint satisfaction
        obj_weight=getattr(config, "obj_weight", 1.0),
        viol_weight=getattr(config, "viol_weight", 1.0),
        viol_threshold=getattr(config, "viol_threshold", 0.0),
        distill_weight=getattr(config, "distill_weight", 0.0),
    )

    utils.train(
        components,
        loss_fn,
        loader_train,
        loader_val,
        lr,
        penalty_growth,
        patience=getattr(config, "patience", 20),
        warmup=getattr(config, "warmup", None),
        validate_every=getattr(config, "validate_every", 125),
        train_eval_batches=getattr(config, "train_eval_batches", 8),
        lr_anneal=getattr(config, "lr_anneal", False),
        lr_min=getattr(config, "lr_min", 1e-6),
        loader_test=loader_test,
        tensorboard=getattr(config, "tb", False),
        tb_logdir=getattr(config, "tb_logdir", "runs"),
        tb_run_name=f"microgrid_ste_T{T}_pen{penalty_weight}",
    )
    df = evaluate(components, loss_fn, model, loader_test, project, config)

    tag = "ste"
    suffix = "-g" if penalty_growth else ("-p" if project else "")
    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_dir / f"mg_{tag}{penalty_weight}_T{T}{suffix}.csv")


def evaluate(components, loss_fn, model, loader_test, project, config):
    """
    Evaluate first N test cases by:
    - forward: params -> x -> x_rnd
    - optional projection: update x to reduce constraint violation
    - write x_rnd back to Pyomo model variables (excluding p_grid which is derived)
    - compute objective and violation from the Pyomo model
    - optional solver baseline solve + final comparison summary
    """
    if project:
        from src.postprocess.project import gradientProjection
        # projection target is "x" (continuous), but we want rounding to regenerate x_rnd during projection
        # pre_components: [concat, smap], post_components: [rnd]
        proj = gradientProjection([components[0], components[1]], [components[2]], loss_fn, "x",
                                  max_iters=getattr(config, "proj_iters", 200),
                                  step_size=getattr(config, "proj_step", 1e-2),
                                  decay=getattr(config, "proj_decay", 1.0),
                                  obj_guard_weight=getattr(config, "proj_obj_guard", 0.0),
                                  viol_tol=getattr(config, "proj_viol_tol", 1e-6),
                                  early_stop_patience=getattr(config, "proj_early_stop_patience", 20),
                                  min_viol_improve=getattr(config, "proj_min_viol_improve", 1e-8),
                                  restarts=getattr(config, "proj_restarts", 1),
                                  restart_noise_std=getattr(config, "proj_restart_noise", 0.0),
                                  tail_steps=getattr(config, "proj_tail_steps", 0),
                                  tail_weight=getattr(config, "proj_tail_weight", 1.0),
                                  end_discharge_steps=getattr(config, "proj_end_discharge_steps", 0),
                                  end_discharge_scale=getattr(config, "proj_end_discharge_scale", 1.0))

    components.eval()
    compare_solver = getattr(config, "compare_solver", False)

    constraint_labels = _microgrid_constraint_labels(model.horizon)
    policy_viol_count = np.zeros(len(constraint_labels), dtype=int)
    solver_viol_count = np.zeros(len(constraint_labels), dtype=int)

    params_list, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    obj_before_balance_vals, obj_after_balance_vals = [], []
    no_proj_objvals, no_proj_mean_viols, no_proj_num_viols = [], [], []
    select_proj_count, select_no_proj_count = 0, 0
    solver_objvals, solver_mean_viols, solver_max_viols = [], [], []
    solver_num_viols, solver_elapseds, obj_gaps, solver_errors = [], [], [], []

    if compare_solver:
        print(
            "Solver baseline enabled: "
            f"timelimit={getattr(config, 'solver_time_limit', 60.0)}s, "
            f"tee={getattr(config, 'solver_tee', False)}"
        )

    def _objective_from_x_numpy(x_vec, params):
        """Compute objective directly from flattened x (no feasibility repair)."""
        s = model.x_slices
        p_grid_buy = x_vec[s["p_grid_buy"]]
        p_grid_sell = x_vec[s["p_grid_sell"]]
        p_gen = x_vec[s["p_gen"]]
        s_load = x_vec[s["s_load"]]
        u_gen = x_vec[s["u_gen"]]

        price_buy = np.asarray(params["price_buy"], dtype=float)
        price_sell = np.asarray(params["price_sell"], dtype=float)

        elec_cost = np.sum(price_buy * p_grid_buy - price_sell * p_grid_sell)
        gen_cost = np.sum(model.gen_quad * (p_gen ** 2) + model.gen_lin * p_gen + model.gen_on_cost * u_gen)
        shed_cost = np.sum(model.load_shed_penalty * s_load)
        return float(elec_cost + gen_cost + shed_cost)

    def _evaluate_candidate_x(x_raw, params):
        """
        Evaluate one policy candidate with both pre/post balance objective diagnostics.
        Returns: (x_balanced, obj_pre_balance, obj_post_balance, obj_pyomo, viol_arr)
        """
        x_local = np.asarray(x_raw, dtype=float).copy()
        # Always apply a cheap deterministic repair before objective/violation eval.
        x_local = _fast_hard_repair_x_numpy(x_local, model, soc0=params["soc0"])
        T = model.horizon
        s = model.x_slices

        obj_pre_balance = _objective_from_x_numpy(x_local, params)
        x_balanced = _enforce_power_balance_in_x_numpy(
            x_local,
            s,
            np.asarray(params["load"], dtype=float),
            np.asarray(params["pv"], dtype=float),
            p_grid_buy_max=model.p_grid_buy_max,
            p_grid_sell_max=model.p_grid_sell_max,
        )
        x_balanced = _economic_post_balance_repair_x_numpy(x_balanced, model)
        obj_post_balance = _objective_from_x_numpy(x_balanced, params)

        # Reconstruct SOC by dynamics (no longer part of x; it's 9T now)
        soc = _reconstruct_soc_from_x_numpy(
            x_balanced,
            s,
            T,
            params["soc0"],
            model.eta_ch,
            model.eta_dis,
        )

        def set_series(varname):
            sl = s[varname]
            for t in range(T):
                model.vars[varname][t].value = float(x_balanced[sl.start + t])

        # Set all variables except 'soc' (which is computed, not learned)
        for varname in ["p_grid_buy", "p_grid_sell", "p_gen", "p_ch", "p_dis", "s_load", "u_gen", "u_ch", "u_dis"]:
            set_series(varname)

        # Set SOC directly from reconstruction
        for t in range(T):
            model.vars["soc"][t].value = float(soc[t])

        # p_grid is excluded from flattened x, but Pyomo constraints use it directly.
        buy_sl = s["p_grid_buy"]
        sell_sl = s["p_grid_sell"]
        for t in range(T):
            p_grid_val = float(x_balanced[buy_sl.start + t] - x_balanced[sell_sl.start + t])
            model.vars["p_grid"][t].value = p_grid_val

        _, obj_pyomo = model.get_val()
        viol_arr = np.asarray(model.cal_violation(), dtype=float)
        return x_balanced, obj_pre_balance, obj_post_balance, float(obj_pyomo), viol_arr

    def _viol_summary_from_x_numpy(x_raw, params):
        """
        Fast violation estimate (with deterministic balance back-substitution) used for
        conditional projection trigger.
        """
        _, _, _, _, viol_arr = _evaluate_candidate_x(x_raw, params)
        return float(np.mean(viol_arr)), int(np.sum(viol_arr > 1e-6))

    # Contrast #2: evaluate all test samples when requested.
    N = len(loader_test.dataset) if getattr(config, "eval_all_test", False) else min(100, len(loader_test.dataset))
    if N < len(loader_test.dataset):
        print(f"[Eval] Using subset: {N}/{len(loader_test.dataset)} test samples.")
    else:
        print(f"[Eval] Using all test samples: {N}.")
    pbar = tqdm(range(N), desc="Eval+Solver" if compare_solver else "Eval")
    for i in pbar:
        # build datapoint dict (shape [1,T] etc.)
        dp = {k: torch.unsqueeze(loader_test.dataset.datadict[k][i], 0).to("cuda")
              for k in ["load", "pv", "price_buy", "price_sell", "soc0"]}
        dp["name"] = "test"

        tick = time.time()
        with torch.no_grad():
            for comp in components:
                dp.update(comp(dp))

        # Keep pre-projection policy output for no-projection contrast.
        x_rnd_no_proj = dp["x_rnd"].detach().cpu().numpy().reshape(-1).copy()

        if project:
            # Conditional projection: only run projection when violation exceeds threshold.
            trigger_mean = getattr(config, "proj_trigger_mean_viol", 1e-3)
            no_proj_mean_viol_est, _ = _viol_summary_from_x_numpy(
                x_rnd_no_proj,
                {
                    "load": dp["load"].detach().cpu().numpy().tolist()[0],
                    "pv": dp["pv"].detach().cpu().numpy().tolist()[0],
                    "price_buy": dp["price_buy"].detach().cpu().numpy().tolist()[0],
                    "price_sell": dp["price_sell"].detach().cpu().numpy().tolist()[0],
                    "soc0": float(dp["soc0"].detach().cpu().numpy().reshape(-1)[0]),
                },
            )
            if no_proj_mean_viol_est > trigger_mean:
                # projection uses autograd internally
                proj(dp)

        tock = time.time()

        # store params for csv readability
        params_list.append({
            "load": dp["load"].detach().cpu().numpy().tolist()[0],
            "pv": dp["pv"].detach().cpu().numpy().tolist()[0],
            "price_buy": dp["price_buy"].detach().cpu().numpy().tolist()[0],
            "price_sell": dp["price_sell"].detach().cpu().numpy().tolist()[0],
            "soc0": float(dp["soc0"].detach().cpu().numpy().reshape(-1)[0]),
        })

        # assign params to Pyomo model
        model.set_param_val({
            "load": np.array(params_list[-1]["load"]),
            "pv": np.array(params_list[-1]["pv"]),
            "price_buy": np.array(params_list[-1]["price_buy"]),
            "price_sell": np.array(params_list[-1]["price_sell"]),
            "soc0": params_list[-1]["soc0"],
        })

        # projected candidate (main policy metric)
        x_rnd_proj = dp["x_rnd"].detach().cpu().numpy().reshape(-1)
        x_proj, obj_pre_balance_proj, obj_post_balance_proj, objval_proj, viol_arr_proj = _evaluate_candidate_x(
            x_rnd_proj,
            params_list[-1],
        )

        # no-projection contrast on the same sample
        x_no_proj, obj_pre_balance_no_proj, obj_post_balance_no_proj, obj_no_proj, viol_no_proj = _evaluate_candidate_x(
            x_rnd_no_proj,
            params_list[-1],
        )

        # Candidate selection for final reported policy metrics.
        select_mode = getattr(config, "policy_select", "best")
        feas_guard = float(getattr(config, "policy_feas_guard", 1e-4))
        if not project:
            select_mode = "no_proj"

        proj_num_viol = int(np.sum(viol_arr_proj > 1e-6))
        proj_mean_viol = float(np.mean(viol_arr_proj))
        no_proj_num_viol = int(np.sum(viol_no_proj > 1e-6))
        no_proj_mean_viol = float(np.mean(viol_no_proj))

        if select_mode == "proj":
            x, obj_pre_balance, obj_post_balance, objval, viol_arr = (
                x_proj,
                obj_pre_balance_proj,
                obj_post_balance_proj,
                objval_proj,
                viol_arr_proj,
            )
            select_proj_count += 1
        elif select_mode == "no_proj":
            x, obj_pre_balance, obj_post_balance, objval, viol_arr = (
                x_no_proj,
                obj_pre_balance_no_proj,
                obj_post_balance_no_proj,
                obj_no_proj,
                viol_no_proj,
            )
            select_no_proj_count += 1
        else:
            # best: prioritize feasibility, then objective.
            proj_feas = proj_num_viol == 0 and proj_mean_viol <= feas_guard
            no_proj_feas = no_proj_num_viol == 0 and no_proj_mean_viol <= feas_guard
            if proj_feas and not no_proj_feas:
                pick_proj = True
            elif no_proj_feas and not proj_feas:
                pick_proj = False
            elif proj_feas and no_proj_feas:
                pick_proj = objval_proj <= obj_no_proj
            else:
                if abs(proj_mean_viol - no_proj_mean_viol) > 1e-12:
                    pick_proj = proj_mean_viol < no_proj_mean_viol
                else:
                    pick_proj = objval_proj <= obj_no_proj

            if pick_proj:
                x, obj_pre_balance, obj_post_balance, objval, viol_arr = (
                    x_proj,
                    obj_pre_balance_proj,
                    obj_post_balance_proj,
                    objval_proj,
                    viol_arr_proj,
                )
                select_proj_count += 1
            else:
                x, obj_pre_balance, obj_post_balance, objval, viol_arr = (
                    x_no_proj,
                    obj_pre_balance_no_proj,
                    obj_post_balance_no_proj,
                    obj_no_proj,
                    viol_no_proj,
                )
                select_no_proj_count += 1

        if len(viol_arr) != len(policy_viol_count):
            # Fallback for future constraint set changes while keeping stats available.
            if len(policy_viol_count) != 0:
                print(
                    f"[Warn] Constraint count mismatch: labels={len(policy_viol_count)}, "
                    f"actual={len(viol_arr)}. Switching to generic labels."
                )
            constraint_labels = [f"cons_{k + 1}" for k in range(len(viol_arr))]
            policy_viol_count = np.zeros(len(viol_arr), dtype=int)
            solver_viol_count = np.zeros(len(viol_arr), dtype=int)

        policy_viol_count += (viol_arr > 1e-6).astype(int)

        # collect outputs
        sols.append(x.tolist())
        objvals.append(float(objval))
        obj_before_balance_vals.append(float(obj_pre_balance))
        obj_after_balance_vals.append(float(obj_post_balance))
        mean_viols.append(float(np.mean(viol_arr)))
        max_viols.append(float(np.max(viol_arr)))
        num_viols.append(int(np.sum(viol_arr > 1e-6)))
        elapseds.append(tock - tick)
        no_proj_objvals.append(float(obj_no_proj))
        no_proj_mean_viols.append(float(np.mean(viol_no_proj)))
        no_proj_num_viols.append(int(np.sum(viol_no_proj > 1e-6)))

        # optional solver baseline on the same scenario for comparison
        if getattr(config, "compare_solver", False):
            solver_tick = time.time()
            model.timelimit = getattr(config, "solver_time_limit", 60.0)
            try:
                _, solver_obj = model.solve(tee=getattr(config, "solver_tee", False))
                solver_viol = model.cal_violation()
                solver_viol_arr = np.asarray(solver_viol, dtype=float)
                if len(solver_viol_arr) == len(solver_viol_count):
                    solver_viol_count += (solver_viol_arr > 1e-6).astype(int)
                solver_elapsed = time.time() - solver_tick

                solver_objvals.append(float(solver_obj) if solver_obj is not None else np.nan)
                solver_mean_viols.append(float(np.mean(solver_viol_arr)))
                solver_max_viols.append(float(np.max(solver_viol_arr)))
                solver_num_viols.append(int(np.sum(solver_viol_arr > 1e-6)))
                solver_elapseds.append(float(solver_elapsed))
                solver_errors.append("")

                if solver_obj is not None and objval is not None and abs(float(solver_obj)) > 1e-9:
                    gap = (float(objval) - float(solver_obj)) / abs(float(solver_obj)) * 100.0
                else:
                    gap = np.nan
                obj_gaps.append(gap)

                # show online comparison progress without overwhelming logs
                pbar.set_postfix({
                    "gap%": f"{gap:.2f}" if not np.isnan(gap) else "nan",
                    "t_pol(ms)": f"{(tock - tick) * 1000:.2f}",
                    "t_sol(ms)": f"{solver_elapsed * 1000:.2f}",
                })
            except Exception as e:
                solver_objvals.append(np.nan)
                solver_mean_viols.append(np.nan)
                solver_max_viols.append(np.nan)
                solver_num_viols.append(np.nan)
                solver_elapseds.append(np.nan)
                obj_gaps.append(np.nan)
                solver_errors.append(f"{type(e).__name__}: {e}")
                pbar.set_postfix({"solver": "error"})

    df = pd.DataFrame({
        "Param": params_list,
        "Sol": sols,
        "Obj Val": objvals,
        "Obj Before Balance": obj_before_balance_vals,
        "Obj After Balance": obj_after_balance_vals,
        "Mean Violation": mean_viols,
        "Max Violation": max_viols,
        "Num Violations": num_viols,
        "NoProj Obj Val": no_proj_objvals,
        "NoProj Mean Violation": no_proj_mean_viols,
        "NoProj Num Violations": no_proj_num_viols,
        "Elapsed Time": elapseds,
    })
    df["Proj Obj Delta"] = df["Obj Val"] - df["NoProj Obj Val"]
    df["Balance Obj Delta"] = df["Obj After Balance"] - df["Obj Before Balance"]

    if getattr(config, "compare_solver", False):
        df["Solver Obj Val"] = solver_objvals
        df["Solver Mean Violation"] = solver_mean_viols
        df["Solver Max Violation"] = solver_max_viols
        df["Solver Num Violations"] = solver_num_viols
        df["Solver Elapsed Time"] = solver_elapseds
        df["Obj Gap (%)"] = obj_gaps
        df["Solver Error"] = solver_errors
        df["Speedup (solver/policy)"] = df["Solver Elapsed Time"] / np.maximum(df["Elapsed Time"], 1e-6)

        # print final comparison summary
        valid = df["Solver Obj Val"].notna() & df["Obj Val"].notna()
        if valid.any():
            print("=== Policy vs Solver Summary ===")
            print(f"Valid compared cases: {int(valid.sum())}/{len(df)}")
            print(
                "Policy Obj mean/median: "
                f"{df.loc[valid, 'Obj Val'].mean():.4f} / {df.loc[valid, 'Obj Val'].median():.4f}"
            )
            print(
                "Solver Obj mean/median: "
                f"{df.loc[valid, 'Solver Obj Val'].mean():.4f} / {df.loc[valid, 'Solver Obj Val'].median():.4f}"
            )
            print(
                "Gap% mean/median: "
                f"{df.loc[valid, 'Obj Gap (%)'].mean():.4f} / {df.loc[valid, 'Obj Gap (%)'].median():.4f}"
            )
            print(
                "Policy/Solver time mean (ms): "
                f"{df.loc[valid, 'Elapsed Time'].mean() * 1000:.3f} / "
                f"{df.loc[valid, 'Solver Elapsed Time'].mean() * 1000:.3f}"
            )
            print(
                "Speedup mean/median (solver/policy): "
                f"{df.loc[valid, 'Speedup (solver/policy)'].mean():.2f} / "
                f"{df.loc[valid, 'Speedup (solver/policy)'].median():.2f}"
            )
            solver_fail = int((df["Solver Error"].astype(str).str.len() > 0).sum())
            print(f"Solver failed cases: {solver_fail}")

    print(df.describe(include="all"))
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print(
        "Policy candidate selection counts (proj / no_proj): "
        f"{select_proj_count} / {select_no_proj_count}"
    )

    # Contrast #1: projection vs no projection on identical test cases.
    if project and len(df) > 0:
        print("=== Projection Contrast (same cases) ===")
        print(
            "NoProj Obj mean/median: "
            f"{df['NoProj Obj Val'].mean():.4f} / {df['NoProj Obj Val'].median():.4f}"
        )
        print(
            "Proj Obj mean/median: "
            f"{df['Obj Val'].mean():.4f} / {df['Obj Val'].median():.4f}"
        )
        print(
            "Proj - NoProj Obj delta mean/median: "
            f"{df['Proj Obj Delta'].mean():.4f} / {df['Proj Obj Delta'].median():.4f}"
        )

    # Contrast #3: objective before vs after deterministic power-balance back-substitution.
    if len(df) > 0:
        print("=== Balance Back-Substitution Contrast ===")
        print(
            "Obj(before balance) mean/median: "
            f"{df['Obj Before Balance'].mean():.4f} / {df['Obj Before Balance'].median():.4f}"
        )
        print(
            "Obj(after balance) mean/median: "
            f"{df['Obj After Balance'].mean():.4f} / {df['Obj After Balance'].median():.4f}"
        )
        print(
            "After - Before delta mean/median: "
            f"{df['Balance Obj Delta'].mean():.4f} / {df['Balance Obj Delta'].median():.4f}"
        )

    if len(policy_viol_count) > 0:
        top_idx = int(np.argmax(policy_viol_count))
        print("=== Most Violated Constraint (Policy) ===")
        print(f"{constraint_labels[top_idx]} -> {int(policy_viol_count[top_idx])} cases")

        # Also print top-5 for quick diagnosis.
        topk = min(5, len(policy_viol_count))
        order = np.argsort(-policy_viol_count)[:topk]
        print("Top-5 Policy violation counts:")
        for idx in order:
            print(f"  {constraint_labels[int(idx)]}: {int(policy_viol_count[int(idx)])}")

    if compare_solver and len(solver_viol_count) > 0:
        top_idx = int(np.argmax(solver_viol_count))
        print("=== Most Violated Constraint (Solver) ===")
        print(f"{constraint_labels[top_idx]} -> {int(solver_viol_count[top_idx])} cases")

    # Save constraint-level violation frequency statistics.
    if len(constraint_labels) > 0:
        result_dir = Path(__file__).resolve().parents[1] / "result"
        result_dir.mkdir(parents=True, exist_ok=True)

        method = getattr(config, "method", "unk")
        penalty = getattr(config, "penalty", "na")
        horizon = getattr(config, "horizon", model.horizon)
        suffix = "-g" if getattr(config, "penalty_growth", False) else ("-p" if getattr(config, "project", False) else "")

        policy_stats = pd.DataFrame({
            "Constraint": constraint_labels,
            "Policy Violation Count": policy_viol_count.astype(int),
            "Policy Violation Rate": policy_viol_count.astype(float) / max(len(df), 1),
        })

        if compare_solver and len(solver_viol_count) == len(policy_viol_count):
            policy_stats["Solver Violation Count"] = solver_viol_count.astype(int)
            policy_stats["Solver Violation Rate"] = solver_viol_count.astype(float) / max(len(df), 1)

        policy_stats = policy_stats.sort_values("Policy Violation Count", ascending=False)
        stats_path = result_dir / f"mg_constraint_stats_{method}{penalty}_T{horizon}{suffix}.csv"
        policy_stats.to_csv(stats_path, index=False)
        print(f"[ConstraintStats] Saved: {stats_path}")

    return df
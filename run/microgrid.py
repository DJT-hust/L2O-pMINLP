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
    hlayers_sol = config.hlayers_sol
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

    # solution map: (params) -> x (continuous candidate, length nx=10T)
    func = nm.modules.blocks.MLP(
        insize=param_dim,
        outsize=model.nx,
        bias=True,
        linear_map=nm.slim.maps["linear"],
        nonlin=nn.ReLU,
        hsizes=[hsize] * hlayers_sol,
    )

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
    )

    utils.train(
        components,
        loss_fn,
        loader_train,
        loader_val,
        lr,
        penalty_growth,
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
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project

    model = msMicrogrid(horizon=T, timelimit=1000)

    param_keys = ["load", "pv", "price_buy", "price_sell", "soc0"]
    param_dim = 4 * T + 1

    func = nm.modules.blocks.MLP(
        insize=param_dim,
        outsize=model.nx,
        bias=True,
        linear_map=nm.slim.maps["linear"],
        nonlin=nn.ReLU,
        hsizes=[hsize] * hlayers_sol,
    )

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
    )

    utils.train(
        components,
        loss_fn,
        loader_train,
        loader_val,
        lr,
        penalty_growth,
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
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project

    model = msMicrogrid(horizon=T, timelimit=1000)

    param_keys = ["load", "pv", "price_buy", "price_sell", "soc0"]
    param_dim = 4 * T + 1

    func = nm.modules.blocks.MLP(
        insize=param_dim,
        outsize=model.nx,
        bias=True,
        linear_map=nm.slim.maps["linear"],
        nonlin=nn.ReLU,
        hsizes=[hsize] * hlayers_sol,
    )

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
    )

    utils.train(
        components,
        loss_fn,
        loader_train,
        loader_val,
        lr,
        penalty_growth,
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
    """
    if project:
        from src.postprocess.project import gradientProjection
        # projection target is "x" (continuous), but we want rounding to regenerate x_rnd during projection
        # pre_components: [concat, smap], post_components: [rnd]
        proj = gradientProjection([components[0], components[1]], [components[2]], loss_fn, "x",
                                  max_iters=getattr(config, "proj_iters", 200),
                                  step_size=getattr(config, "proj_step", 1e-2),
                                  decay=getattr(config, "proj_decay", 1.0))

    components.eval()

    params_list, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []

    # only evaluate a subset to keep runtime sane
    N = min(100, len(loader_test.dataset))
    for i in tqdm(range(N), desc="Eval"):
        # build datapoint dict (shape [1,T] etc.)
        dp = {k: torch.unsqueeze(loader_test.dataset.datadict[k][i], 0).to("cuda")
              for k in ["load", "pv", "price_buy", "price_sell", "soc0"]}
        dp["name"] = "test"

        tick = time.time()
        with torch.no_grad():
            for comp in components:
                dp.update(comp(dp))

        if project:
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

        # write x_rnd to Pyomo vars
        x = dp["x_rnd"].detach().cpu().numpy().reshape(-1)
        T = model.horizon
        s = model.x_slices

        def set_series(varname):
            sl = s[varname]
            for t in range(T):
                model.vars[varname][t].value = float(x[sl.start + t])

        for varname in ["p_grid_buy", "p_grid_sell", "p_gen", "p_ch", "p_dis", "soc", "s_load", "u_gen", "u_ch", "u_dis"]:
            set_series(varname)

        # p_grid is excluded from flattened x, but Pyomo constraints use it directly.
        # Set it explicitly to avoid evaluating constraints with uninitialized VarData.
        buy_sl = s["p_grid_buy"]
        sell_sl = s["p_grid_sell"]
        for t in range(T):
            p_grid_val = float(x[buy_sl.start + t] - x[sell_sl.start + t])
            model.vars["p_grid"][t].value = p_grid_val

        # evaluate objective and violations from math model
        xval, objval = model.get_val()
        viol = model.cal_violation()

        # collect outputs
        sols.append(x.tolist())
        objvals.append(objval)
        mean_viols.append(float(np.mean(viol)))
        max_viols.append(float(np.max(viol)))
        num_viols.append(int(np.sum(np.array(viol) > 1e-6)))
        elapseds.append(tock - tick)

    df = pd.DataFrame({
        "Param": params_list,
        "Sol": sols,
        "Obj Val": objvals,
        "Mean Violation": mean_viols,
        "Max Violation": max_viols,
        "Num Violations": num_viols,
        "Elapsed Time": elapseds,
    })

    print(df.describe(include="all"))
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    return df
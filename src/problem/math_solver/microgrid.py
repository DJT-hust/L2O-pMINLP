"""
Parametric Mixed Integer Quadratic Microgrid Scheduling

NOTE:
- This math_solver model keeps p_grid as a Pyomo variable (for equality constraint convenience),
  but the learning framework can use a flattened vector x that EXCLUDES p_grid.
- We add a consistent flattening scheme (x_slices) + index sets (bin_ind/int_ind) for pMINLP rounding.
"""

import numpy as np
from pyomo import environ as pe
import gurobipy as gp

try:
    from .abc_solver import abcParamSolver
except ImportError:
    from abc_solver import abcParamSolver


class microgrid(abcParamSolver):
    def __init__(
        self,
        horizon,
        solver="gurobi",
        p_gen_max=2.0,
        p_ch_max=1.0,
        p_dis_max=1.0,
        soc_min=0.1,
        soc_max=4.0,
        p_grid_buy_max=3.0,
        p_grid_sell_max=3.0,
        eta_ch=0.95,
        eta_dis=0.95,
        gen_quad=0.06,
        gen_lin=0.1,
        gen_on_cost=0.2,
        load_shed_penalty=10.0,
        timelimit=None,
    ):
        super().__init__(timelimit=timelimit, solver="gurobi")

        # Keep useful scalars for downstream (loss / indexing)
        self.horizon = int(horizon)
        self.p_gen_max = float(p_gen_max)
        self.p_ch_max = float(p_ch_max)
        self.p_dis_max = float(p_dis_max)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.p_grid_buy_max = float(p_grid_buy_max)
        self.p_grid_sell_max = float(p_grid_sell_max)
        self.eta_ch = float(eta_ch)
        self.eta_dis = float(eta_dis)
        self.gen_quad = float(gen_quad)
        self.gen_lin = float(gen_lin)
        self.gen_on_cost = float(gen_on_cost)
        self.load_shed_penalty = float(load_shed_penalty)

        # create model
        m = pe.ConcreteModel()
        m.T = pe.RangeSet(0, self.horizon - 1)

        # mutable parameters (parametric scenario input)
        m.p_load = pe.Param(m.T, default=0.0, mutable=True)
        m.pv = pe.Param(m.T, default=0.0, mutable=True)
        m.price_buy = pe.Param(m.T, default=0.0, mutable=True)
        m.price_sell = pe.Param(m.T, default=0.0, mutable=True)
        m.soc0 = pe.Param(default=(soc_min + soc_max) / 2, mutable=True)

        # decision variables
        m.p_grid_buy = pe.Var(m.T, domain=pe.NonNegativeReals, bounds=(0.0, p_grid_buy_max))
        m.p_grid_sell = pe.Var(m.T, domain=pe.NonNegativeReals, bounds=(0.0, p_grid_sell_max))
        m.p_grid = pe.Var(m.T, domain=pe.Reals)

        m.p_gen = pe.Var(m.T, domain=pe.NonNegativeReals)
        m.p_ch = pe.Var(m.T, domain=pe.NonNegativeReals)
        m.p_dis = pe.Var(m.T, domain=pe.NonNegativeReals)
        m.soc = pe.Var(m.T, domain=pe.Reals)
        m.s_load = pe.Var(m.T, domain=pe.NonNegativeReals)

        m.u_gen = pe.Var(m.T, domain=pe.Binary)
        m.u_ch = pe.Var(m.T, domain=pe.Binary)
        m.u_dis = pe.Var(m.T, domain=pe.Binary)

        # objective
        elec_cost = sum(m.price_buy[t] * m.p_grid_buy[t] - m.price_sell[t] * m.p_grid_sell[t] for t in m.T)
        gen_cost = sum(gen_quad * m.p_gen[t] ** 2 + gen_lin * m.p_gen[t] + gen_on_cost * m.u_gen[t] for t in m.T)
        shed_cost = sum(load_shed_penalty * m.s_load[t] for t in m.T)
        m.obj = pe.Objective(sense=pe.minimize, expr=elec_cost + gen_cost + shed_cost)

        # constraints
        m.cons = pe.ConstraintList()
        for t in m.T:
            # net exchange variable from buy/sell split
            m.cons.add(m.p_grid[t] == m.p_grid_buy[t] - m.p_grid_sell[t])

            # power balance: p_gen + p_dis - p_ch + p_grid + pv + s_load = load
            m.cons.add(
                m.p_gen[t] + m.p_dis[t] - m.p_ch[t] + m.p_grid[t] + m.pv[t] + m.s_load[t] == m.p_load[t]
            )

            # generator bounds linked with commitment
            m.cons.add(m.p_gen[t] <= p_gen_max * m.u_gen[t])

            # battery charge/discharge bounds and exclusivity
            m.cons.add(m.p_ch[t] <= p_ch_max * m.u_ch[t])
            m.cons.add(m.p_dis[t] <= p_dis_max * m.u_dis[t])
            m.cons.add(m.u_ch[t] + m.u_dis[t] <= 1)

            # SOC bounds
            m.cons.add(m.soc[t] >= soc_min)
            m.cons.add(m.soc[t] <= soc_max)

            # SOC dynamics
            if t == 0:
                m.cons.add(m.soc[t] == m.soc0 + eta_ch * m.p_ch[t] - (1.0 / eta_dis) * m.p_dis[t])
            else:
                m.cons.add(m.soc[t] == m.soc[t - 1] + eta_ch * m.p_ch[t] - (1.0 / eta_dis) * m.p_dis[t])

        # set attributes for abcParamSolver
        self.model = m
        self.params = {
            "load": m.p_load,
            "pv": m.pv,
            "price_buy": m.price_buy,
            "price_sell": m.price_sell,
            "soc0": m.soc0,
        }
        self.vars = {
            # NOTE: p_grid exists in the math solver model, but will be EXCLUDED from flattened x used by learning
            "p_grid": m.p_grid,
            "p_grid_buy": m.p_grid_buy,
            "p_grid_sell": m.p_grid_sell,
            "p_gen": m.p_gen,
            "p_ch": m.p_ch,
            "p_dis": m.p_dis,
            "soc": m.soc,
            "s_load": m.s_load,
            "u_gen": m.u_gen,
            "u_ch": m.u_ch,
            "u_dis": m.u_dis,
        }
        self.cons = m.cons

        # ------------------------------------------------------------------
        # pMINLP flattening scheme for learning (EXCLUDES p_grid and soc)
        # SOC is derived from charge/discharge dynamics via _reconstruct_soc()
        # x = [p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis]
        # each block has length T; total nx = 9T
        # ------------------------------------------------------------------
        T = self.horizon
        self.nx = 9 * T
        self.x_slices = {
            "p_grid_buy": slice(0 * T, 1 * T),
            "p_grid_sell": slice(1 * T, 2 * T),
            "p_gen": slice(2 * T, 3 * T),
            "p_ch": slice(3 * T, 4 * T),
            "p_dis": slice(4 * T, 5 * T),
            "s_load": slice(5 * T, 6 * T),
            "u_gen": slice(6 * T, 7 * T),
            "u_ch": slice(7 * T, 8 * T),
            "u_dis": slice(8 * T, 9 * T),
        }
        bin_inds = list(range(6 * T, 9 * T))  # u_gen,u_ch,u_dis
        # roundModel expects dict keyed by variable name, e.g. {"x": indices}
        self.bin_ind = {"x": bin_inds}
        self.int_ind = {"x": bin_inds}  # binaries are also integers


if __name__ == "__main__":
    try:
        from src.utlis import ms_test_solve
    except ModuleNotFoundError:
        import os
        import sys

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from src.utlis import ms_test_solve

    horizon = 24
    rng = np.random.RandomState(17)

    # generate one scenario
    load = 1.4 + 0.4 * rng.rand(horizon)
    pv = 0.6 * rng.rand(horizon)
    price_buy = 0.5 + 0.2 * rng.rand(horizon)
    price_sell = 0.2 + 0.1 * rng.rand(horizon)
    soc0 = 1.5

    params = {
        "load": load,
        "pv": pv,
        "price_buy": price_buy,
        "price_sell": price_sell,
        "soc0": soc0,
    }

    model = microgrid(horizon=horizon)

    print("======================================================")
    print("Solve microgrid scheduling MIQP:")
    model.set_param_val(params)
    solvals, _ = ms_test_solve(model, tee=True)

    print()
    print("======================================================")
    print("Warm start:")
    model.set_param_val(params)
    model.set_warm_start(solvals)
    ms_test_solve(model, tee=True)

    print()
    print("======================================================")
    print("Solve penalty problem:")
    model_pen = model.penalty(100)
    model_pen.set_param_val(params)
    ms_test_solve(model_pen)

    print()
    print("======================================================")
    print("Solve relaxed problem:")
    model_rel = model.relax()
    model_rel.set_param_val(params)
    ms_test_solve(model_rel)
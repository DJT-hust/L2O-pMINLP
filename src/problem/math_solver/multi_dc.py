"""
Parametric Multi-Data-Center scheduling with IEEE 33-bus LinDistFlow constraints.

Key model blocks:
- Thermal + external-grid purchase + renewable use
- Interactive routing and migration
- Batch placement/processing and migration
- Data-center activation/switching
- IEEE 33-bus radial LinDistFlow network constraints
- Root bus interaction with external grid
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from pyomo import environ as pe
from pyomo import opt as po

try:
    from .abc_solver import abcParamSolver
except ImportError:
    from abc_solver import abcParamSolver


class multiDC(abcParamSolver):
    def __init__(
        self,
        horizon: int,
        num_dc: int = 3,
        num_regions: int = 3,
        num_jobs: int = 4,
        num_nodes_per_dc: Sequence[int] | int | None = None,
        cpu_per_node: Sequence[float] | float | None = None,
        solver: str = "gurobi_persistent",
        timelimit: float | None = None,
        p_th_max: float = 8.0,
        p_grid_max: float = 15.0,
        q_grid_abs_max: float = 12.0,
        thermal_quad: float = 0.0,
        thermal_lin: float = 0.2,
        thermal_on_cost: float = 0.1,
        dc_idle_power: Sequence[float] | None = None,
        dc_cpu_cap: Sequence[float] | None = None,
        alpha_interactive: float = 0.9,
        alpha_batch: float = 0.6,
        phi_interactive: Sequence[float] | None = None,
        latency_mask: np.ndarray | None = None,
        release_times: Sequence[int] | None = None,
        deadlines: Sequence[int] | None = None,
        y_init: Sequence[float] | None = None,
        switch_cost: Sequence[float] | None = None,
        mig_cost_interactive: Sequence[float] | None = None,
        mig_cost_batch: Sequence[float] | None = None,
        curtail_cost: Sequence[float] | None = None,
        late_penalty: Sequence[float] | None = None,
        dc_bus_map: Sequence[int] | None = None,
        v_min_sq: float = -1.0e3,
        v_max_sq: float = 1.0e3,
        dc_reactive_factor: float = 0.23,
        line_flow_abs_max: float = 20.0,
    ):
        super().__init__(timelimit=timelimit, solver=solver)

        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)

        if num_nodes_per_dc is None:
            self.num_nodes_per_dc = np.ones(self.num_dc, dtype=int)
        elif np.isscalar(num_nodes_per_dc):
            self.num_nodes_per_dc = np.full(self.num_dc, int(num_nodes_per_dc), dtype=int)
        else:
            self.num_nodes_per_dc = np.asarray(num_nodes_per_dc, dtype=int)
        if self.num_nodes_per_dc.shape[0] != self.num_dc:
            raise ValueError("num_nodes_per_dc length must equal num_dc.")
        if np.any(self.num_nodes_per_dc <= 0):
            raise ValueError("num_nodes_per_dc must be positive for every DC.")
        self.max_nodes_per_dc = int(np.max(self.num_nodes_per_dc))

        self.p_th_max = float(p_th_max)
        self.p_grid_max = float(p_grid_max)
        self.q_grid_abs_max = float(q_grid_abs_max)
        self.thermal_quad = float(thermal_quad)
        self.thermal_lin = float(thermal_lin)
        self.thermal_on_cost = float(thermal_on_cost)
        self.alpha_interactive = float(alpha_interactive)
        self.alpha_batch = float(alpha_batch)
        self.v_min_sq = float(v_min_sq)
        self.v_max_sq = float(v_max_sq)
        self.dc_reactive_factor = float(dc_reactive_factor)

        self.dc_idle_power = np.asarray(dc_idle_power if dc_idle_power is not None else [0.25] * self.num_dc, dtype=float)

        if cpu_per_node is None:
            if dc_cpu_cap is not None:
                cap_tmp = np.asarray(dc_cpu_cap, dtype=float)
                self.cpu_per_node = cap_tmp / np.maximum(self.num_nodes_per_dc.astype(float), 1.0)
            else:
                self.cpu_per_node = np.asarray([1.0] * self.num_dc, dtype=float)
        elif np.isscalar(cpu_per_node):
            self.cpu_per_node = np.full(self.num_dc, float(cpu_per_node), dtype=float)
        else:
            self.cpu_per_node = np.asarray(cpu_per_node, dtype=float)
        if self.cpu_per_node.shape[0] != self.num_dc:
            raise ValueError("cpu_per_node length must equal num_dc.")

        if dc_cpu_cap is None:
            self.dc_cpu_cap = self.cpu_per_node * self.num_nodes_per_dc.astype(float)
        else:
            self.dc_cpu_cap = np.asarray(dc_cpu_cap, dtype=float)
            if self.dc_cpu_cap.shape[0] != self.num_dc:
                raise ValueError("dc_cpu_cap length must equal num_dc.")
        self.phi_interactive = np.asarray(
            phi_interactive if phi_interactive is not None else [1.0] * self.num_regions,
            dtype=float,
        )

        if latency_mask is None:
            latency_mask = np.ones((self.num_regions, self.num_dc), dtype=float)
        self.latency_mask = np.asarray(latency_mask, dtype=float)

        if release_times is None:
            release_times = np.linspace(0, max(0, self.horizon // 2), self.num_jobs, dtype=int)
        if deadlines is None:
            deadlines = np.clip(np.asarray(release_times, dtype=int) + max(1, self.horizon // 3), 0, self.horizon - 1)
        self.release_times = np.asarray(release_times, dtype=int)
        self.deadlines = np.asarray(deadlines, dtype=int)

        self.y_init = np.asarray(y_init if y_init is not None else [1.0] * self.num_dc, dtype=float)
        self.switch_cost = np.asarray(switch_cost if switch_cost is not None else [0.1] * self.num_dc, dtype=float)
        self.node_init = np.zeros((self.num_dc, self.max_nodes_per_dc), dtype=float)
        for d in range(self.num_dc):
            init_val = 1.0 if float(self.y_init[d]) >= 0.5 else 0.0
            self.node_init[d, : int(self.num_nodes_per_dc[d])] = init_val
        self.mig_cost_interactive = np.asarray(
            mig_cost_interactive if mig_cost_interactive is not None else [0.08] * self.num_regions,
            dtype=float,
        )
        self.mig_cost_batch = np.asarray(mig_cost_batch if mig_cost_batch is not None else [0.05] * self.num_jobs, dtype=float)
        self.curtail_cost = np.asarray(curtail_cost if curtail_cost is not None else [0.02] * self.num_dc, dtype=float)
        self.late_penalty = np.asarray(late_penalty if late_penalty is not None else [2.0] * self.num_jobs, dtype=float)

        # IEEE 33-bus feeder constants (plain text, 1-based buses in source data).
        # Per-bus nominal active/reactive demand (MW / MVar, converted from kW/kVar scale).
        self.ieee33_p_bus_mw = np.array(
            [
                0.00, 0.10, 0.09, 0.12, 0.06, 0.06, 0.20, 0.20, 0.06, 0.06, 0.045,
                0.06, 0.06, 0.12, 0.06, 0.06, 0.06, 0.09, 0.09, 0.09, 0.09, 0.09,
                0.09, 0.42, 0.42, 0.06, 0.06, 0.06, 0.12, 0.20, 0.15, 0.21, 0.06,
            ],
            dtype=float,
        )
        self.ieee33_q_bus_mvar = np.array(
            [
                0.00, 0.06, 0.04, 0.08, 0.03, 0.02, 0.10, 0.10, 0.02, 0.02, 0.03,
                0.035, 0.035, 0.08, 0.01, 0.02, 0.02, 0.04, 0.04, 0.04, 0.04, 0.04,
                0.05, 0.20, 0.20, 0.025, 0.025, 0.02, 0.07, 0.60, 0.07, 0.10, 0.04,
            ],
            dtype=float,
        )

        # Branches in radial order (from, to), and line parameters (r, x).
        b_from_1b = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 19, 20, 21, 3, 23, 24, 6, 26, 27, 28, 29, 30, 31, 32],
            dtype=int,
        )
        b_to_1b = np.array(
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            dtype=int,
        )
        # Scale line impedances for MW-level power units used in this model.
        z_scale = 0.01
        self.branch_r = z_scale * np.array(
            [
                0.0922, 0.4930, 0.3660, 0.3811, 0.8190, 0.1872, 1.7114, 1.0300,
                1.0440, 0.1966, 0.3744, 1.4680, 0.5416, 0.5910, 0.7463, 1.2890,
                0.7320, 0.1640, 1.5042, 0.4095, 0.7089, 0.4512, 0.8980, 0.8960,
                0.2030, 0.2842, 1.0590, 0.8042, 0.5075, 0.9744, 0.3105, 0.3410,
            ],
            dtype=float,
        )
        self.branch_x = z_scale * np.array(
            [
                0.0470, 0.2511, 0.1864, 0.1941, 0.7070, 0.6188, 1.2351, 0.7400,
                0.7400, 0.0650, 0.1238, 1.1550, 0.7129, 0.5260, 0.5450, 1.7210,
                0.5740, 0.1565, 1.3554, 0.4784, 0.9373, 0.3083, 0.7091, 0.7011,
                0.1034, 0.1447, 0.9337, 0.7006, 0.2585, 0.9630, 0.3619, 0.5302,
            ],
            dtype=float,
        )

        self.branch_from = (b_from_1b - 1).astype(int)
        self.branch_to = (b_to_1b - 1).astype(int)
        self.num_bus = int(self.ieee33_p_bus_mw.shape[0])
        self.num_branch = int(self.branch_from.shape[0])
        self.root_bus = 0

        if dc_bus_map is None:
            # 1-based buses [18, 25, 33] -> 0-based indices.
            dc_bus_map = [17, 24, 32]
        self.dc_bus_map = np.asarray(dc_bus_map, dtype=int)
        if self.dc_bus_map.shape[0] != self.num_dc:
            raise ValueError("dc_bus_map length must equal num_dc.")

        self.line_flow_abs_max = np.full(self.num_branch, float(line_flow_abs_max), dtype=float)

        p_sum = max(1e-6, float(np.sum(self.ieee33_p_bus_mw)))
        q_sum = max(1e-6, float(np.sum(self.ieee33_q_bus_mvar)))
        self.p_base_share = self.ieee33_p_bus_mw / p_sum
        self.q_base_share = self.ieee33_q_bus_mvar / q_sum

        # Tree adjacency helpers.
        self.parent_edge_of_bus = np.full(self.num_bus, -1, dtype=int)
        self.children_edges_of_bus = [[] for _ in range(self.num_bus)]
        for e in range(self.num_branch):
            i = int(self.branch_from[e])
            j = int(self.branch_to[e])
            self.parent_edge_of_bus[j] = e
            self.children_edges_of_bus[i].append(e)

        m = pe.ConcreteModel()
        m.T = pe.RangeSet(0, self.horizon - 1)
        m.D = pe.RangeSet(0, self.num_dc - 1)
        m.K = pe.RangeSet(0, self.max_nodes_per_dc - 1)
        m.R = pe.RangeSet(0, self.num_regions - 1)
        m.J = pe.RangeSet(0, self.num_jobs - 1)
        m.Tp = pe.RangeSet(1, self.horizon - 1) if self.horizon > 1 else pe.RangeSet(0, -1)
        m.N = pe.RangeSet(0, self.num_bus - 1)
        m.E = pe.RangeSet(0, self.num_branch - 1)

        # Scenario parameters (mutable).
        m.base_load = pe.Param(m.T, default=0.0, mutable=True)
        m.price = pe.Param(m.T, default=0.0, mutable=True)
        m.renew_avail = pe.Param(m.D, m.T, default=0.0, mutable=True)
        m.interactive = pe.Param(m.R, m.T, default=0.0, mutable=True)
        m.batch_work = pe.Param(m.J, default=0.0, mutable=True)

        # Fixed model parameters.
        m.lat_ok = pe.Param(m.R, m.D, initialize={(r, d): float(self.latency_mask[r, d]) for r in range(self.num_regions) for d in range(self.num_dc)})
        m.node_active = pe.Param(
            m.D,
            m.K,
            initialize={
                (d, k): 1.0 if k < int(self.num_nodes_per_dc[d]) else 0.0
                for d in range(self.num_dc)
                for k in range(self.max_nodes_per_dc)
            },
        )

        # Decision variables.
        m.p_th = pe.Var(m.T, domain=pe.NonNegativeReals)
        m.u_th = pe.Var(m.T, domain=pe.Binary)
        m.p_grid = pe.Var(m.T, domain=pe.NonNegativeReals, bounds=(0.0, self.p_grid_max))
        m.q_grid = pe.Var(m.T, domain=pe.Reals, bounds=(-self.q_grid_abs_max, self.q_grid_abs_max))

        # Root import and branch DistFlow state.
        m.p_sub = pe.Var(m.T, domain=pe.NonNegativeReals, bounds=(0.0, self.p_grid_max + self.p_th_max))
        m.q_sub = pe.Var(m.T, domain=pe.Reals, bounds=(-self.q_grid_abs_max, self.q_grid_abs_max))
        m.pf = pe.Var(m.E, m.T, domain=pe.Reals)
        m.qf = pe.Var(m.E, m.T, domain=pe.Reals)
        m.v = pe.Var(m.N, m.T, domain=pe.Reals, bounds=(self.v_min_sq, self.v_max_sq))

        m.ren_use = pe.Var(m.D, m.T, domain=pe.NonNegativeReals)
        m.y = pe.Var(m.D, m.T, domain=pe.Binary)
        m.on = pe.Var(m.D, m.K, m.T, domain=pe.Binary)

        m.xI = pe.Var(m.R, m.D, m.T, domain=pe.Binary)
        m.z = pe.Var(m.J, m.D, m.T, domain=pe.Binary)
        m.f = pe.Var(m.J, m.D, m.T, domain=pe.NonNegativeReals)
        m.delta = pe.Var(m.J, domain=pe.Binary)

        # Auxiliary variables for linearized switching/migration costs.
        m.eta = pe.Var(m.D, m.T, domain=pe.NonNegativeReals)
        m.nu = pe.Var(m.D, m.K, m.T, domain=pe.NonNegativeReals)
        m.mI = pe.Var(m.R, m.D, m.D, m.Tp, domain=pe.Binary)
        m.mB = pe.Var(m.J, m.D, m.D, m.Tp, domain=pe.Binary)

        # Objective terms.
        c_th = sum(
            self.thermal_quad * (m.p_th[t] ** 2) + self.thermal_lin * m.p_th[t] + self.thermal_on_cost * m.u_th[t]
            for t in m.T
        )
        c_grid = sum(m.price[t] * m.p_grid[t] for t in m.T)
        c_sw = sum(float(self.switch_cost[d]) * m.eta[d, t] for d in m.D for t in m.T)
        c_sw_node = sum(
            float(self.switch_cost[d]) * m.nu[d, k, t]
            for d in range(self.num_dc)
            for k in range(int(self.num_nodes_per_dc[d]))
            for t in range(self.horizon)
        )
        c_mig_i = sum(
            float(self.mig_cost_interactive[r]) * m.mI[r, d, dp, t]
            for r in m.R
            for d in m.D
            for dp in m.D
            for t in m.Tp
            if d != dp
        )
        c_mig_b = sum(
            float(self.mig_cost_batch[j]) * m.mB[j, d, dp, t]
            for j in m.J
            for d in m.D
            for dp in m.D
            for t in m.Tp
            if d != dp
        )
        c_cur = sum(float(self.curtail_cost[d]) * (m.renew_avail[d, t] - m.ren_use[d, t]) for d in m.D for t in m.T)
        c_late = sum(float(self.late_penalty[j]) * (1.0 - m.delta[j]) for j in m.J)

        m.obj = pe.Objective(
            sense=pe.minimize,
            expr=c_th + c_grid + c_sw + c_sw_node + c_mig_i + c_mig_b + c_cur + c_late,
        )

        m.cons = pe.ConstraintList()

        def _dc_power_expr(d, t):
            li_expr = sum(self.phi_interactive[r] * m.interactive[r, t] * m.xI[r, d, t] for r in m.R)
            batch_expr = sum(m.f[j, d, t] for j in m.J)
            return self.dc_idle_power[d] * m.y[d, t] + self.alpha_interactive * li_expr + self.alpha_batch * batch_expr

        def _node_count_expr(d, t):
            return sum(m.on[d, k, t] for k in range(int(self.num_nodes_per_dc[d])))

        # Core power / generation constraints.
        for t in m.T:
            m.cons.add(m.p_th[t] <= self.p_th_max * m.u_th[t])
            for d in m.D:
                m.cons.add(m.ren_use[d, t] <= m.renew_avail[d, t])
                # Keep renewable consumption local to DC demand (no feeder export variable in this model).
                m.cons.add(m.ren_use[d, t] <= _dc_power_expr(d, t))

        # Interactive assignment + latency feasibility.
        for r in m.R:
            for t in m.T:
                m.cons.add(sum(m.xI[r, d, t] for d in m.D) == 1.0)
                for d in m.D:
                    m.cons.add(m.xI[r, d, t] <= m.lat_ok[r, d])

        # Batch location and processing windows.
        for j in m.J:
            rel = int(self.release_times[j])
            ddl = int(self.deadlines[j])
            # Hard completion regime: every batch job must be completed.
            m.cons.add(m.delta[j] == 1.0)
            for t in m.T:
                if rel <= int(t) <= ddl:
                    m.cons.add(sum(m.z[j, d, t] for d in m.D) == 1.0)
                else:
                    m.cons.add(sum(m.z[j, d, t] for d in m.D) == 0.0)
                for d in m.D:
                    m.cons.add(m.f[j, d, t] <= self.dc_cpu_cap[d] * m.z[j, d, t])

            m.cons.add(
                sum(m.f[j, d, t] for d in m.D for t in m.T if rel <= int(t) <= ddl)
                >= m.batch_work[j]
            )

        # Activation consistency and per-DC capacity.
        for d in m.D:
            for t in m.T:
                m.cons.add(_node_count_expr(d, t) >= m.y[d, t])
                m.cons.add(_node_count_expr(d, t) <= float(self.num_nodes_per_dc[d]) * m.y[d, t])
                for k in range(int(self.num_nodes_per_dc[d]), self.max_nodes_per_dc):
                    m.cons.add(m.on[d, k, t] == 0.0)

                li_expr = sum(self.phi_interactive[r] * m.interactive[r, t] * m.xI[r, d, t] for r in m.R)
                batch_expr = sum(m.f[j, d, t] for j in m.J)

                m.cons.add(li_expr + batch_expr <= self.cpu_per_node[d] * _node_count_expr(d, t))

                for r in m.R:
                    m.cons.add(m.xI[r, d, t] <= m.y[d, t])
                for j in m.J:
                    m.cons.add(m.z[j, d, t] <= m.y[d, t])

        # DistFlow: voltage and branch balance over IEEE33 radial network.
        for t in m.T:
            # Root bus interaction with external grid and thermal generation.
            m.cons.add(m.p_sub[t] == m.p_grid[t] + m.p_th[t])
            m.cons.add(m.q_sub[t] == m.q_grid[t])

            # Root voltage fixed at nominal value.
            m.cons.add(m.v[self.root_bus, t] == 1.0)

            # Branch flow bounds and voltage drop equations.
            for e in m.E:
                i = int(self.branch_from[e])
                j = int(self.branch_to[e])
                m.cons.add(m.pf[e, t] <= self.line_flow_abs_max[e])
                m.cons.add(m.pf[e, t] >= -self.line_flow_abs_max[e])
                m.cons.add(m.qf[e, t] <= self.line_flow_abs_max[e])
                m.cons.add(m.qf[e, t] >= -self.line_flow_abs_max[e])
                m.cons.add(m.v[j, t] == m.v[i, t] - 2.0 * (self.branch_r[e] * m.pf[e, t] + self.branch_x[e] * m.qf[e, t]))

            # Node power balances for all non-root buses.
            for n in range(1, self.num_bus):
                pe_idx = int(self.parent_edge_of_bus[n])
                if pe_idx < 0:
                    continue
                child_edges = self.children_edges_of_bus[n]

                p_non_dc = self.p_base_share[n] * m.base_load[t]
                q_non_dc = self.q_base_share[n] * m.base_load[t]

                dc_terms = [d for d in range(self.num_dc) if int(self.dc_bus_map[d]) == n]
                if len(dc_terms) > 0:
                    p_dc_n = sum(_dc_power_expr(d, t) - m.ren_use[d, t] for d in dc_terms)
                    q_dc_n = sum(self.dc_reactive_factor * _dc_power_expr(d, t) for d in dc_terms)
                else:
                    p_dc_n = 0.0
                    q_dc_n = 0.0

                p_load_n = p_non_dc + p_dc_n
                q_load_n = q_non_dc + q_dc_n

                m.cons.add(m.pf[pe_idx, t] == p_load_n + sum(m.pf[e2, t] for e2 in child_edges))
                m.cons.add(m.qf[pe_idx, t] == q_load_n + sum(m.qf[e2, t] for e2 in child_edges))

            # Root branch aggregation to substation exchange.
            root_children = self.children_edges_of_bus[self.root_bus]
            m.cons.add(m.p_sub[t] == sum(m.pf[e, t] for e in root_children))
            m.cons.add(m.q_sub[t] == sum(m.qf[e, t] for e in root_children))

        # Switching and migration linearizations.
        for d in m.D:
            m.cons.add(m.eta[d, 0] >= m.y[d, 0] - float(self.y_init[d]))
            m.cons.add(m.eta[d, 0] >= float(self.y_init[d]) - m.y[d, 0])
            for t in m.Tp:
                m.cons.add(m.eta[d, t] >= m.y[d, t] - m.y[d, t - 1])
                m.cons.add(m.eta[d, t] >= m.y[d, t - 1] - m.y[d, t])

            for k in range(int(self.num_nodes_per_dc[d])):
                m.cons.add(m.nu[d, k, 0] >= m.on[d, k, 0] - float(self.node_init[d, k]))
                m.cons.add(m.nu[d, k, 0] >= float(self.node_init[d, k]) - m.on[d, k, 0])
                for t in m.Tp:
                    m.cons.add(m.nu[d, k, t] >= m.on[d, k, t] - m.on[d, k, t - 1])
                    m.cons.add(m.nu[d, k, t] >= m.on[d, k, t - 1] - m.on[d, k, t])

        for t in m.Tp:
            for r in m.R:
                for d in m.D:
                    for dp in m.D:
                        if d == dp:
                            continue
                        m.cons.add(m.mI[r, d, dp, t] >= m.xI[r, d, t - 1] + m.xI[r, dp, t] - 1.0)
            for j in m.J:
                for d in m.D:
                    for dp in m.D:
                        if d == dp:
                            continue
                        m.cons.add(m.mB[j, d, dp, t] >= m.z[j, d, t - 1] + m.z[j, dp, t] - 1.0)

        self.model = m
        self.params = {
            "base_load": m.base_load,
            "price": m.price,
            "renew_avail": m.renew_avail,
            "interactive_demand": m.interactive,
            "batch_work": m.batch_work,
        }
        self.vars = {
            "p_th": m.p_th,
            "u_th": m.u_th,
            "p_grid": m.p_grid,
            "q_grid": m.q_grid,
            "p_sub": m.p_sub,
            "q_sub": m.q_sub,
            "pf": m.pf,
            "qf": m.qf,
            "v": m.v,
            "ren_use": m.ren_use,
            "y": m.y,
            "on": m.on,
            "xI": m.xI,
            "z": m.z,
            "f": m.f,
            "delta": m.delta,
            "eta": m.eta,
            "nu": m.nu,
            "mI": m.mI,
            "mB": m.mB,
        }
        self.cons = m.cons

        # Flattening for learning (excluding aux eta/mI/mB).
        T, D, R, J = self.horizon, self.num_dc, self.num_regions, self.num_jobs
        offset = 0

        def alloc(name: str, length: int):
            nonlocal offset
            sl = slice(offset, offset + length)
            offset += length
            return sl

        self.x_slices = {
            "p_th": alloc("p_th", T),
            "u_th": alloc("u_th", T),
            "p_grid": alloc("p_grid", T),
            "ren_use": alloc("ren_use", D * T),
            "y": alloc("y", D * T),
            "xI": alloc("xI", R * D * T),
            "z": alloc("z", J * D * T),
            "f": alloc("f", J * D * T),
            "delta": alloc("delta", J),
        }
        self.nx = offset

        bin_idx = []
        for k in ["u_th", "y", "xI", "z", "delta"]:
            sl = self.x_slices[k]
            bin_idx.extend(list(range(sl.start, sl.stop)))
        self.bin_ind = {"x": bin_idx}
        self.int_ind = {"x": bin_idx}

    def set_param_val(self, param_dict):
        """Set mutable params with shape-safe assignment for 1D/2D arrays."""
        for key, val in param_dict.items():
            param = self.params[key]
            arr = np.asarray(val)

            if isinstance(param, pe.Param) and (not param.is_indexed()):
                param.set_value(float(arr.reshape(-1)[0]))
                continue

            for idx in param:
                if isinstance(idx, tuple):
                    param[idx].set_value(float(arr[idx]))
                else:
                    param[idx].set_value(float(arr[idx]))

        self._has_warm_start = False

    def _create_solver(self):
        """Create a solver that matches self.solver (avoid hard dependency on gurobi)."""
        opt = po.SolverFactory(self.solver)
        if self.timelimit is not None:
            if self.solver in {"gurobi", "gurobi_persistent"}:
                opt.options["TimeLimit"] = self.timelimit
            elif self.solver == "scip":
                opt.options["limits/time"] = self.timelimit
            elif self.solver == "highs":
                opt.options["time_limit"] = self.timelimit
            elif self.solver == "cbc":
                opt.options["seconds"] = self.timelimit
            elif self.solver == "glpk":
                # glpk has no unified timelimit option exposed by pyomo in all versions.
                pass
        return opt

    def solve(self, tee=False, keepfiles=False, logfile=None):
        """Handle persistent and non-persistent solvers in a unified way."""
        self._refresh_integrality_indices()

        if not self._has_warm_start:
            for var in self.model.component_objects(pe.Var, active=True):
                for index in var:
                    var[index].value = None

        self.opt = self._create_solver()

        if self.solver.endswith("_persistent"):
            if hasattr(self.opt, "set_instance"):
                self.opt.set_instance(self.model)
            self.res = self.opt.solve(tee=tee, keepfiles=keepfiles, logfile=logfile)
        else:
            self.res = self.opt.solve(self.model, tee=tee, keepfiles=keepfiles, logfile=logfile)

        self._has_warm_start = False
        return self.get_val()


if __name__ == "__main__":
    rng = np.random.RandomState(7)
    model = multiDC(horizon=24 * 12, num_dc=3, num_regions=3, num_jobs=4, timelimit=10, solver="gurobi")
    params = {
        "base_load": 2.0 + 0.2 * rng.rand(24 * 12),
        "price": 6.0 + 0.5 * rng.rand(24 * 12),
        "renew_avail": 0.8 * rng.rand(3, 24 * 12),
        "interactive_demand": 0.5 * rng.rand(3, 24 * 12),
        "batch_work": 1.0 + 0.5 * rng.rand(4),
    }
    model.set_param_val(params)
    sol, obj = model.solve(tee=False)
    print("solved", sol is not None, "obj", obj)

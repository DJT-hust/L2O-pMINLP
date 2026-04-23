"""
Neuromancer-style penalty loss for simplified multi-data-center scheduling.
"""

from __future__ import annotations
import torch
from torch import nn


class penaltyLoss(nn.Module):
    def __init__(
        self,
        input_keys,
        horizon: int,
        num_dc: int,
        num_regions: int,
        num_jobs: int,
        x_slices: dict,
        latency_mask,
        release_times,
        deadlines,
        p_th_max: float,
        p_grid_max: float,
        dc_idle_power,
        dc_cpu_cap,
        phi_interactive,
        alpha_interactive: float,
        alpha_batch: float,
        thermal_quad: float,
        thermal_lin: float,
        thermal_on_cost: float,
        switch_cost,
        mig_cost_interactive,
        mig_cost_batch,
        curtail_cost,
        late_penalty,
        y_init,
        p_base_share,
        q_base_share,
        branch_from,
        branch_to,
        branch_r,
        branch_x,
        parent_edge_of_bus,
        children_edges_of_bus,
        dc_bus_map,
        line_flow_abs_max,
        v_min_sq,
        v_max_sq,
        dc_reactive_factor,
        num_nodes_per_dc=None,
        mu_service=None,
        v_latency_factor=None,
        p_idle_it=None,
        p_peak_it=None,
        p_other_dc=None,
        p_cool_slope=None,
        p_cool_bias=None,
        h_cool_max=None,
        r_th=None,
        c_th=None,
        theta_min=None,
        theta_max=None,
        theta_init=None,
        tout_profile=None,
        dc_power_factor=None,
        penalty_weight: float = 40.0,
        eq_weight: float = 1.0,
        obj_weight: float = 1.0,
        viol_weight: float = 1.0,
        output_key: str = "loss",
    ):
        super().__init__()
        assert len(input_keys) == 6, "Expect input_keys=['base_load','price','renew_avail','interactive_demand','batch_work','x_key']"

        self.base_load_key, self.price_key, self.renew_key, self.interactive_key, self.batch_work_key, self.x_key = input_keys

        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)
        self.x_slices = x_slices

        self.p_th_max = float(p_th_max)
        self.p_grid_max = float(p_grid_max)
        self.alpha_interactive = float(alpha_interactive)
        self.alpha_batch = float(alpha_batch)

        self.thermal_quad = float(thermal_quad)
        self.thermal_lin = float(thermal_lin)
        self.thermal_on_cost = float(thermal_on_cost)

        self.penalty_weight = float(penalty_weight)
        self.eq_weight = float(eq_weight)
        self.obj_weight = float(obj_weight)
        self.viol_weight = float(viol_weight)
        self.output_key = output_key

        # Register constants as buffers for GPU-safe arithmetic.
        self.register_buffer("latency_mask", torch.as_tensor(latency_mask, dtype=torch.float32))
        self.register_buffer("release_times", torch.as_tensor(release_times, dtype=torch.long))
        self.register_buffer("deadlines", torch.as_tensor(deadlines, dtype=torch.long))
        self.register_buffer("dc_idle_power", torch.as_tensor(dc_idle_power, dtype=torch.float32))
        self.register_buffer("dc_cpu_cap", torch.as_tensor(dc_cpu_cap, dtype=torch.float32))
        if mu_service is None:
            mu_service = [0.95] * self.num_dc
        if v_latency_factor is None:
            v_latency_factor = [2.0] * self.num_dc
        if p_idle_it is None:
            p_idle_it = [0.10] * self.num_dc
        if p_peak_it is None:
            p_peak_it = [0.35] * self.num_dc
        if p_other_dc is None:
            p_other_dc = [0.08] * self.num_dc
        if p_cool_slope is None:
            p_cool_slope = [0.80] * self.num_dc
        if p_cool_bias is None:
            p_cool_bias = [0.02] * self.num_dc
        if h_cool_max is None:
            h_cool_max = [3.0] * self.num_dc
        if r_th is None:
            r_th = [0.6] * self.num_dc
        if c_th is None:
            c_th = [3.2] * self.num_dc
        if theta_min is None:
            theta_min = [18.0] * self.num_dc
        if theta_max is None:
            theta_max = [28.0] * self.num_dc
        if theta_init is None:
            theta_init = [23.0] * self.num_dc
        if tout_profile is None:
            tgrid = torch.linspace(0.0, 2.0 * torch.pi, steps=self.horizon + 1, dtype=torch.float32)[:-1]
            tout_profile = 26.0 + 4.0 * torch.sin(tgrid - torch.pi / 3.0)
            tout_profile = tout_profile.view(1, -1).repeat(self.num_dc, 1)
        if dc_power_factor is None:
            dc_power_factor = [0.95] * self.num_dc
        self.register_buffer("mu_service", torch.as_tensor(mu_service, dtype=torch.float32))
        self.register_buffer("v_latency_factor", torch.as_tensor(v_latency_factor, dtype=torch.float32))
        self.register_buffer("p_idle_it", torch.as_tensor(p_idle_it, dtype=torch.float32))
        self.register_buffer("p_peak_it", torch.as_tensor(p_peak_it, dtype=torch.float32))
        self.register_buffer("p_other_dc", torch.as_tensor(p_other_dc, dtype=torch.float32))
        self.register_buffer("p_cool_slope", torch.as_tensor(p_cool_slope, dtype=torch.float32))
        self.register_buffer("p_cool_bias", torch.as_tensor(p_cool_bias, dtype=torch.float32))
        self.register_buffer("h_cool_max", torch.as_tensor(h_cool_max, dtype=torch.float32))
        self.register_buffer("r_th", torch.as_tensor(r_th, dtype=torch.float32))
        self.register_buffer("c_th", torch.as_tensor(c_th, dtype=torch.float32))
        self.register_buffer("theta_min", torch.as_tensor(theta_min, dtype=torch.float32))
        self.register_buffer("theta_max", torch.as_tensor(theta_max, dtype=torch.float32))
        self.register_buffer("theta_init", torch.as_tensor(theta_init, dtype=torch.float32))
        self.register_buffer("tout_profile", torch.as_tensor(tout_profile, dtype=torch.float32))
        self.register_buffer("dc_power_factor", torch.as_tensor(dc_power_factor, dtype=torch.float32))
        self.register_buffer("phi_interactive", torch.as_tensor(phi_interactive, dtype=torch.float32))
        self.register_buffer("switch_cost", torch.as_tensor(switch_cost, dtype=torch.float32))
        if num_nodes_per_dc is None:
            num_nodes_per_dc = [1.0] * self.num_dc
        self.register_buffer("num_nodes_per_dc", torch.as_tensor(num_nodes_per_dc, dtype=torch.float32))
        self.register_buffer("mig_cost_interactive", torch.as_tensor(mig_cost_interactive, dtype=torch.float32))
        self.register_buffer("mig_cost_batch", torch.as_tensor(mig_cost_batch, dtype=torch.float32))
        self.register_buffer("curtail_cost", torch.as_tensor(curtail_cost, dtype=torch.float32))
        self.register_buffer("late_penalty", torch.as_tensor(late_penalty, dtype=torch.float32))
        self.register_buffer("y_init", torch.as_tensor(y_init, dtype=torch.float32))
        self.register_buffer("p_base_share", torch.as_tensor(p_base_share, dtype=torch.float32))
        self.register_buffer("q_base_share", torch.as_tensor(q_base_share, dtype=torch.float32))
        self.register_buffer("branch_from", torch.as_tensor(branch_from, dtype=torch.long))
        self.register_buffer("branch_to", torch.as_tensor(branch_to, dtype=torch.long))
        self.register_buffer("branch_r", torch.as_tensor(branch_r, dtype=torch.float32))
        self.register_buffer("branch_x", torch.as_tensor(branch_x, dtype=torch.float32))
        self.register_buffer("parent_edge_of_bus", torch.as_tensor(parent_edge_of_bus, dtype=torch.long))
        self.register_buffer("dc_bus_map", torch.as_tensor(dc_bus_map, dtype=torch.long))
        self.register_buffer("line_flow_abs_max", torch.as_tensor(line_flow_abs_max, dtype=torch.float32))

        self.v_min_sq = float(v_min_sq)
        self.v_max_sq = float(v_max_sq)
        self.dc_reactive_factor = float(dc_reactive_factor)
        self.num_bus = int(self.p_base_share.shape[0])
        self.num_branch = int(self.branch_from.shape[0])
        self.root_bus = 0

        self._children_edges = children_edges_of_bus

        edge_of_child = torch.full((self.num_bus,), -1, dtype=torch.long)
        for e in range(self.num_branch):
            child = int(self.branch_to[e].item())
            edge_of_child[child] = e
        self.register_buffer("edge_of_child", edge_of_child)

        topo_nodes = []
        for b in range(self.num_bus):
            if b == self.root_bus:
                continue
            topo_nodes.append(b)
        self.topo_nodes = topo_nodes
        self.rev_topo_nodes = list(reversed(topo_nodes))

        mig_mask = torch.ones((self.num_dc, self.num_dc), dtype=torch.float32)
        mig_mask.fill_diagonal_(0.0)
        self.register_buffer("mig_mask", mig_mask)

        valid = torch.zeros((self.num_jobs, self.horizon), dtype=torch.float32)
        for j in range(self.num_jobs):
            rel = int(self.release_times[j].item())
            ddl = int(self.deadlines[j].item())
            valid[j, rel : ddl + 1] = 1.0
        self.register_buffer("job_valid_mask", valid)

    def _slice(self, x: torch.Tensor, key: str):
        sl = self.x_slices[key]
        return x[:, sl]

    def _unpack(self, x: torch.Tensor):
        B = x.shape[0]
        T, D, R, J = self.horizon, self.num_dc, self.num_regions, self.num_jobs

        p_th = self._slice(x, "p_th")
        u_th = self._slice(x, "u_th")
        p_grid = self._slice(x, "p_grid")
        ren_use = self._slice(x, "ren_use").reshape(B, D, T)
        y = self._slice(x, "y").reshape(B, D, T)
        xI = self._slice(x, "xI").reshape(B, R, D, T)
        z = self._slice(x, "z").reshape(B, J, D, T)
        f = self._slice(x, "f").reshape(B, J, D, T)
        delta = self._slice(x, "delta")
        return p_th, u_th, p_grid, ren_use, y, xI, z, f, delta

    def _compute_internal_dc_power(self, li: torch.Tensor, batch_proc: torch.Tensor, y: torch.Tensor):
        # Smooth approximation of PDF internal DC model using predicted workload and activation.
        # li, batch_proc, y: [B, D, T]
        eps = 1.0e-6
        cap_i = torch.clamp(self.mu_service - 1.0 / torch.clamp(self.v_latency_factor, min=eps), min=eps)
        cap_i = cap_i.view(1, self.num_dc, 1)
        mu = torch.clamp(self.mu_service, min=eps).view(1, self.num_dc, 1)
        n_nodes = torch.clamp(self.num_nodes_per_dc, min=1.0).view(1, self.num_dc, 1)

        active_nodes = torch.clamp(y, min=0.0, max=1.0) * n_nodes
        mIM = torch.clamp(li / cap_i, min=0.0)
        mBM = torch.clamp(batch_proc / mu, min=0.0)
        mIR = torch.relu(active_nodes - mIM - mBM)
        mBR = torch.zeros_like(mIR)
        mIP = mIM
        mBP = mBM

        e_i = self.p_idle_it.view(1, self.num_dc, 1)
        e_p = self.p_peak_it.view(1, self.num_dc, 1)
        p_other = self.p_other_dc.view(1, self.num_dc, 1)

        p_it_m = e_i * (mIM + mBM) + (e_p - e_i) / mu * (li + batch_proc) + e_i * (mIM + mBM) / n_nodes
        p_it_h = e_i * (mIR + mBR) + (e_p - e_i) * (mIP + mBP) + e_i * (mIR + mBR) / n_nodes
        p_it = torch.clamp(p_it_m + p_it_h, min=0.0)

        h_raw = torch.clamp(0.6 * (p_it + p_other), min=0.0)
        h_guess = torch.minimum(h_raw, self.h_cool_max.view(1, self.num_dc, 1))
        p_c = self.p_cool_slope.view(1, self.num_dc, 1) * h_guess + self.p_cool_bias.view(1, self.num_dc, 1)
        p_dc = torch.clamp(p_it + p_c + p_other, min=0.0)
        return p_dc

    def cal_obj(self, input_dict):
        x = input_dict[self.x_key]
        price = input_dict[self.price_key]
        renew_avail = input_dict[self.renew_key]

        p_th, u_th, p_grid, ren_use, y, xI, z, f, delta = self._unpack(x)

        c_th = (self.thermal_quad * (p_th ** 2) + self.thermal_lin * p_th + self.thermal_on_cost * u_th).sum(dim=1)
        c_grid = (price * p_grid).sum(dim=1)

        # Switching proxy using absolute differences.
        eta0 = torch.abs(y[:, :, 0] - self.y_init.unsqueeze(0))
        etat = torch.abs(y[:, :, 1:] - y[:, :, :-1]).sum(dim=2)
        c_sw = (self.switch_cost.unsqueeze(0) * (eta0 + etat)).sum(dim=1)
        c_sw_node = (
            self.switch_cost.unsqueeze(0)
            * self.num_nodes_per_dc.unsqueeze(0)
            * (eta0 + etat)
        ).sum(dim=1)

        if self.horizon > 1:
            # Interactive migration proxy.
            x_prev = xI[:, :, :, :-1]  # [B,R,D,T-1]
            x_cur = xI[:, :, :, 1:]    # [B,R,D,T-1]
            mig_i = torch.relu(x_prev.unsqueeze(3) + x_cur.unsqueeze(2) - 1.0)
            mig_i = mig_i * self.mig_mask.view(1, 1, self.num_dc, self.num_dc, 1)
            c_mig_i = (self.mig_cost_interactive.view(1, self.num_regions, 1, 1, 1) * mig_i).sum(dim=(1, 2, 3, 4))

            z_prev = z[:, :, :, :-1]
            z_cur = z[:, :, :, 1:]
            mig_b = torch.relu(z_prev.unsqueeze(3) + z_cur.unsqueeze(2) - 1.0)
            mig_b = mig_b * self.mig_mask.view(1, 1, self.num_dc, self.num_dc, 1)
            c_mig_b = (self.mig_cost_batch.view(1, self.num_jobs, 1, 1, 1) * mig_b).sum(dim=(1, 2, 3, 4))
        else:
            c_mig_i = torch.zeros_like(c_grid)
            c_mig_b = torch.zeros_like(c_grid)

        c_cur = (self.curtail_cost.view(1, self.num_dc, 1) * torch.relu(renew_avail - ren_use)).sum(dim=(1, 2))
        c_late = (self.late_penalty.view(1, self.num_jobs) * (1.0 - delta)).sum(dim=1)

        return c_th + c_grid + c_sw + c_sw_node + c_mig_i + c_mig_b + c_cur + c_late

    def cal_constr_viol(self, input_dict):
        x = input_dict[self.x_key]
        base_load = input_dict[self.base_load_key]
        renew_avail = input_dict[self.renew_key]
        interactive = input_dict[self.interactive_key]
        batch_work = input_dict[self.batch_work_key]

        p_th, u_th, p_grid, ren_use, y, xI, z, f, delta = self._unpack(x)

        relu = torch.relu

        # Equality violations.
        eq_viol = torch.zeros(x.shape[0], device=x.device)

        # Interactive unique assignment.
        assign_sum = xI.sum(dim=2)  # [B,R,T]
        eq_viol = eq_viol + ((assign_sum - 1.0) ** 2).sum(dim=(1, 2))

        # Batch location window constraints.
        z_sum = z.sum(dim=2)  # [B,J,T]
        valid = self.job_valid_mask.view(1, self.num_jobs, self.horizon)
        target = valid  # valid time -> 1, invalid -> 0
        eq_viol = eq_viol + ((z_sum - target) ** 2).sum(dim=(1, 2))
        # Hard completion switch: every batch job must be completed.
        eq_viol = eq_viol + ((delta - 1.0) ** 2).sum(dim=1)

        # Build per-DC power demand from computing states.
        li = (
            self.phi_interactive.view(1, self.num_regions, 1, 1)
            * interactive.unsqueeze(2)
            * xI
        ).sum(dim=1)  # [B,D,T]
        batch_proc = f.sum(dim=1)  # [B,D,T]
        p_dc = self._compute_internal_dc_power(li, batch_proc, y)

        # DistFlow-derived feeder demand from bus loads.
        B = x.shape[0]
        T = self.horizon
        N = self.num_bus
        E = self.num_branch

        p_non_dc = self.p_base_share.view(1, N, 1) * base_load.unsqueeze(1)
        q_non_dc = self.q_base_share.view(1, N, 1) * base_load.unsqueeze(1)

        p_dc_bus = torch.zeros((B, N, T), device=x.device)
        q_dc_bus = torch.zeros((B, N, T), device=x.device)
        ren_bus = torch.zeros((B, N, T), device=x.device)
        for d in range(self.num_dc):
            bus_idx = int(self.dc_bus_map[d].item())
            p_dc_bus[:, bus_idx, :] = p_dc_bus[:, bus_idx, :] + p_dc[:, d, :]
            q_dc_bus[:, bus_idx, :] = q_dc_bus[:, bus_idx, :] + self.dc_reactive_factor * p_dc[:, d, :]
            ren_bus[:, bus_idx, :] = ren_bus[:, bus_idx, :] + ren_use[:, d, :]

        p_load = p_non_dc + p_dc_bus - ren_bus
        q_load = q_non_dc + q_dc_bus

        # Upstream subtree accumulation in radial tree.
        p_subtree = p_load.clone()
        q_subtree = q_load.clone()
        for n in self.rev_topo_nodes:
            p_edge = int(self.parent_edge_of_bus[n].item())
            if p_edge < 0:
                continue
            parent = int(self.branch_from[p_edge].item())
            p_subtree[:, parent, :] = p_subtree[:, parent, :] + p_subtree[:, n, :]
            q_subtree[:, parent, :] = q_subtree[:, parent, :] + q_subtree[:, n, :]

        p_flow = torch.zeros((B, E, T), device=x.device)
        q_flow = torch.zeros((B, E, T), device=x.device)
        for e in range(E):
            child = int(self.branch_to[e].item())
            p_flow[:, e, :] = p_subtree[:, child, :]
            q_flow[:, e, :] = q_subtree[:, child, :]

        # Root power coupling: p_sub = p_th + p_grid = sum(root outgoing branch flows).
        root_children = self._children_edges[self.root_bus]
        p_sub_root = torch.zeros((B, T), device=x.device)
        for e in root_children:
            p_sub_root = p_sub_root + p_flow[:, e, :]
        eq_viol = eq_viol + ((p_th + p_grid - p_sub_root) ** 2).sum(dim=1)

        # Voltage recursion and bounds.
        v = torch.zeros((B, N, T), device=x.device)
        v[:, self.root_bus, :] = 1.0
        for n in self.topo_nodes:
            e = int(self.edge_of_child[n].item())
            if e < 0:
                continue
            parent = int(self.branch_from[e].item())
            v[:, n, :] = v[:, parent, :] - 2.0 * (self.branch_r[e] * p_flow[:, e, :] + self.branch_x[e] * q_flow[:, e, :])

        # Inequality / bounds.
        ineq_viol = torch.zeros_like(eq_viol)
        ineq_viol = ineq_viol + relu(-p_th).sum(dim=1)
        ineq_viol = ineq_viol + relu(-p_grid).sum(dim=1)
        ineq_viol = ineq_viol + relu(-ren_use).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(-f).sum(dim=(1, 2, 3))

        ineq_viol = ineq_viol + relu(p_th - self.p_th_max * u_th).sum(dim=1)
        ineq_viol = ineq_viol + relu(p_grid - self.p_grid_max).sum(dim=1)
        ineq_viol = ineq_viol + relu(ren_use - renew_avail).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(ren_use - p_dc).sum(dim=(1, 2))

        # DistFlow line and voltage limits.
        line_lim = self.line_flow_abs_max.view(1, E, 1)
        ineq_viol = ineq_viol + relu(torch.abs(p_flow) - line_lim).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(torch.abs(q_flow) - line_lim).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(self.v_min_sq - v).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(v - self.v_max_sq).sum(dim=(1, 2))

        # Latency feasibility and activation implications.
        ineq_viol = ineq_viol + relu(xI - self.latency_mask.view(1, self.num_regions, self.num_dc, 1)).sum(dim=(1, 2, 3))
        ineq_viol = ineq_viol + relu(xI - y.unsqueeze(1)).sum(dim=(1, 2, 3))
        ineq_viol = ineq_viol + relu(z - y.unsqueeze(1)).sum(dim=(1, 2, 3))

        # Batch processing bound f <= cap * z.
        ineq_viol = ineq_viol + relu(
            f - self.dc_cpu_cap.view(1, 1, self.num_dc, 1) * z
        ).sum(dim=(1, 2, 3))

        # DC capacity bound LI + batch <= cap * y.
        ineq_viol = ineq_viol + relu(
            li + batch_proc - self.dc_cpu_cap.view(1, self.num_dc, 1) * y
        ).sum(dim=(1, 2))

        # Thermal feasibility proxy: mirror the smooth internal DC temperature dynamics.
        dt_hour = 5.0 / 60.0
        kappa = torch.exp(-dt_hour / torch.clamp(self.r_th * self.c_th, min=1.0e-4)).view(1, self.num_dc, 1)
        h_guess = torch.minimum(
            torch.clamp(0.6 * (p_dc + self.p_other_dc.view(1, self.num_dc, 1)), min=0.0),
            self.h_cool_max.view(1, self.num_dc, 1),
        )
        p_c = self.p_cool_slope.view(1, self.num_dc, 1) * h_guess + self.p_cool_bias.view(1, self.num_dc, 1)
        p_it = torch.clamp(p_dc - p_c - self.p_other_dc.view(1, self.num_dc, 1), min=0.0)
        theta = torch.zeros((x.shape[0], self.num_dc, self.horizon), device=x.device)
        theta_prev = self.theta_init.view(1, self.num_dc)
        for t in range(self.horizon):
            theta[:, :, t] = (
                kappa.squeeze(-1) * theta_prev
                + (1.0 - kappa.squeeze(-1)) * self.tout_profile[:, t].view(1, self.num_dc)
                + self.r_th.view(1, self.num_dc)
                * (1.0 - kappa.squeeze(-1))
                * (p_it[:, :, t] + self.p_other_dc.view(1, self.num_dc) - h_guess[:, :, t])
            )
            theta_prev = theta[:, :, t]
        ineq_viol = ineq_viol + relu(theta - self.theta_max.view(1, self.num_dc, 1)).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(self.theta_min.view(1, self.num_dc, 1) - theta).sum(dim=(1, 2))

        # Radial feeder feasibility proxy (LinDistFlow-style).
        B = x.shape[0]
        N = self.num_bus
        E = self.num_branch
        p_non_dc = self.p_base_share.view(1, N, 1) * base_load.unsqueeze(1)
        q_non_dc = self.q_base_share.view(1, N, 1) * base_load.unsqueeze(1)

        p_dc_bus = torch.zeros((B, N, self.horizon), device=x.device)
        q_dc_bus = torch.zeros((B, N, self.horizon), device=x.device)
        ren_bus = torch.zeros((B, N, self.horizon), device=x.device)
        for d in range(self.num_dc):
            bus_idx = int(self.dc_bus_map[d].item())
            p_dc_bus[:, bus_idx, :] = p_dc_bus[:, bus_idx, :] + p_dc[:, d, :]
            q_dc_bus[:, bus_idx, :] = q_dc_bus[:, bus_idx, :] + self.dc_reactive_factor * p_dc[:, d, :]
            ren_bus[:, bus_idx, :] = ren_bus[:, bus_idx, :] + ren_use[:, d, :]

        p_load = p_non_dc + p_dc_bus - ren_bus
        q_load = q_non_dc + q_dc_bus

        p_subtree = p_load.clone()
        q_subtree = q_load.clone()
        for n in self.rev_topo_nodes:
            p_edge = int(self.parent_edge_of_bus[n].item())
            if p_edge < 0:
                continue
            parent = int(self.branch_from[p_edge].item())
            p_subtree[:, parent, :] = p_subtree[:, parent, :] + p_subtree[:, n, :]
            q_subtree[:, parent, :] = q_subtree[:, parent, :] + q_subtree[:, n, :]

        p_flow = torch.zeros((B, E, self.horizon), device=x.device)
        q_flow = torch.zeros((B, E, self.horizon), device=x.device)
        for e in range(E):
            child = int(self.branch_to[e].item())
            p_flow[:, e, :] = p_subtree[:, child, :]
            q_flow[:, e, :] = q_subtree[:, child, :]

        root_children = self._children_edges[self.root_bus]
        p_sub_root = torch.zeros((B, self.horizon), device=x.device)
        for e in root_children:
            p_sub_root = p_sub_root + p_flow[:, e, :]
        eq_viol = eq_viol + ((p_th + p_grid - p_sub_root) ** 2).sum(dim=1)

        v = torch.zeros((B, N, self.horizon), device=x.device)
        v[:, self.root_bus, :] = 1.0
        for n in self.topo_nodes:
            e = int(self.edge_of_child[n].item())
            if e < 0:
                continue
            parent = int(self.branch_from[e].item())
            v[:, n, :] = v[:, parent, :] - 2.0 * (
                self.branch_r[e] * p_flow[:, e, :] + self.branch_x[e] * q_flow[:, e, :]
            )
        line_lim = self.line_flow_abs_max.view(1, E, 1)
        ineq_viol = ineq_viol + relu(torch.abs(p_flow) - line_lim).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(torch.abs(q_flow) - line_lim).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(self.v_min_sq - v).sum(dim=(1, 2))
        ineq_viol = ineq_viol + relu(v - self.v_max_sq).sum(dim=(1, 2))

        # Batch completion with deadline.
        valid4 = self.job_valid_mask.view(1, self.num_jobs, 1, self.horizon)
        processed = (f * valid4).sum(dim=(2, 3))  # [B,J]
        ineq_viol = ineq_viol + relu(batch_work - processed).sum(dim=1)

        # Keep binary surrogates in [0,1].
        for bvar in [u_th, y.reshape(x.shape[0], -1), xI.reshape(x.shape[0], -1), z.reshape(x.shape[0], -1), delta]:
            ineq_viol = ineq_viol + relu(-bvar).sum(dim=1) + relu(bvar - 1.0).sum(dim=1)

        return self.eq_weight * eq_viol + ineq_viol

    def forward(self, input_dict):
        obj = self.cal_obj(input_dict)
        viol = self.cal_constr_viol(input_dict)
        loss = self.obj_weight * obj + self.penalty_weight * self.viol_weight * viol
        input_dict[self.output_key] = loss.mean()
        return input_dict

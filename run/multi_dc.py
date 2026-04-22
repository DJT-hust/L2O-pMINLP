#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for simplified Multi-DC scheduling.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyomo import environ as pe
from torch import nn
from tqdm import tqdm

from run import utils


def _build_param_concat_node(param_keys):
    import neuromancer as nm

    class ParamConcat(nn.Module):
        def forward(self, *vals):
            return torch.cat(list(vals), dim=-1)

    return nm.system.Node(ParamConcat(), param_keys, ["xi"], name="concat")


def _flatten_param_dim(horizon: int, num_dc: int, num_regions: int, num_jobs: int):
    return horizon + horizon + num_dc * horizon + num_regions * horizon + num_jobs


def _build_smap(config, in_dim: int, out_dim: int):
    from src.func.layer import (
        MultiDCLSTMPolicy,
        MultiDCRNNPolicy,
        MultiDCTCNPolicy,
        MultiDCDualHeadMLPPolicy,
        MultiDCDualHeadTCNPolicy,
        MultiDCMLPPolicy,
        netFC,
    )

    hsize = int(getattr(config, "hsize", 128))
    depth = int(getattr(config, "hlayers_sol", 6))
    smap_arch = str(getattr(config, "smap_arch", "mlp")).lower()
    head_mode = str(getattr(config, "smap_head_mode", "single")).lower()
    use_residual = bool(int(getattr(config, "smap_residual", 0)))
    residual_scale = float(getattr(config, "smap_residual_scale", 0.3))
    inject_feats = bool(int(getattr(config, "constraint_feat_inject", 0)))

    if smap_arch == "lstm":
        return MultiDCLSTMPolicy(
            horizon=int(getattr(config, "horizon")),
            num_dc=int(getattr(config, "num_dc")),
            num_regions=int(getattr(config, "num_regions")),
            num_jobs=int(getattr(config, "num_jobs")),
            out_dim=out_dim,
            hidden_dim=hsize,
            num_layers=int(getattr(config, "lstm_layers", 2)),
            dropout=float(getattr(config, "lstm_dropout", 0.1)),
            constraint_feat_inject=inject_feats,
        )

    if smap_arch == "rnn":
        return MultiDCRNNPolicy(
            horizon=int(getattr(config, "horizon")),
            num_dc=int(getattr(config, "num_dc")),
            num_regions=int(getattr(config, "num_regions")),
            num_jobs=int(getattr(config, "num_jobs")),
            out_dim=out_dim,
            hidden_dim=hsize,
            num_layers=int(getattr(config, "rnn_layers", 2)),
            dropout=float(getattr(config, "rnn_dropout", 0.1)),
            constraint_feat_inject=inject_feats,
        )

    if smap_arch == "tcn":
        if head_mode == "dual":
            return MultiDCDualHeadTCNPolicy(
                horizon=int(getattr(config, "horizon")),
                num_dc=int(getattr(config, "num_dc")),
                num_regions=int(getattr(config, "num_regions")),
                num_jobs=int(getattr(config, "num_jobs")),
                out_dim=out_dim,
                hidden_dim=hsize,
                num_blocks=int(getattr(config, "tcn_blocks", 4)),
                dropout=float(getattr(config, "tcn_dropout", 0.1)),
                residual=use_residual,
                residual_scale=residual_scale,
                constraint_feat_inject=inject_feats,
            )
        return MultiDCTCNPolicy(
            horizon=int(getattr(config, "horizon")),
            num_dc=int(getattr(config, "num_dc")),
            num_regions=int(getattr(config, "num_regions")),
            num_jobs=int(getattr(config, "num_jobs")),
            out_dim=out_dim,
            hidden_dim=hsize,
            num_blocks=int(getattr(config, "tcn_blocks", 4)),
            dropout=float(getattr(config, "tcn_dropout", 0.1)),
            constraint_feat_inject=inject_feats,
        )

    if head_mode == "dual":
        return MultiDCDualHeadMLPPolicy(
            input_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hsize,
            depth=depth,
            dropout=0.2,
            residual=use_residual,
            residual_scale=residual_scale,
            horizon=int(getattr(config, "horizon")),
            num_dc=int(getattr(config, "num_dc")),
            num_regions=int(getattr(config, "num_regions")),
            num_jobs=int(getattr(config, "num_jobs")),
            constraint_feat_inject=inject_feats,
        )
    if inject_feats:
        return MultiDCMLPPolicy(
            horizon=int(getattr(config, "horizon")),
            num_dc=int(getattr(config, "num_dc")),
            num_regions=int(getattr(config, "num_regions")),
            num_jobs=int(getattr(config, "num_jobs")),
            input_dim=in_dim,
            hidden_dim=hsize,
            depth=depth,
            out_dim=out_dim,
            dropout=0.2,
            constraint_feat_inject=True,
        )
    return netFC(input_dim=in_dim, hidden_dims=[hsize] * depth, output_dim=out_dim)


def _build_reduced_x_layout(model):
    """Build reduced decision layout by eliminating one DC column for equality-constrained xI and z."""
    T = int(model.horizon)
    D = int(model.num_dc)
    R = int(model.num_regions)
    J = int(model.num_jobs)
    Dm1 = D - 1

    if Dm1 <= 0:
        raise ValueError("num_dc must be >= 2 for equality-based dimensionality reduction")

    offset = 0

    def alloc(length: int):
        nonlocal offset
        sl = slice(offset, offset + length)
        offset += length
        return sl

    red_slices = {
        "p_th": alloc(T),
        "u_th": alloc(T),
        "p_grid": alloc(T),
        "ren_use": alloc(D * T),
        "y": alloc(D * T),
        "xI_red": alloc(R * Dm1 * T),
        "z_red": alloc(J * Dm1 * T),
        "f": alloc(J * D * T),
        "delta": alloc(J),
    }
    nx_red = offset

    bin_idx = []
    for k in ["u_th", "y", "xI_red", "z_red", "delta"]:
        sl = red_slices[k]
        bin_idx.extend(list(range(sl.start, sl.stop)))

    int_ind = {"x_red": bin_idx}
    bin_ind = {"x_red": bin_idx}
    return red_slices, nx_red, int_ind, bin_ind


class _EqualityReconstructNode(nn.Module):
    """Reconstruct full x from reduced x by enforcing equality constraints for xI and z exactly."""

    def __init__(self, model, red_slices):
        super().__init__()
        self.red_slices = red_slices
        self.T = int(model.horizon)
        self.D = int(model.num_dc)
        self.R = int(model.num_regions)
        self.J = int(model.num_jobs)
        self.Dm1 = self.D - 1
        self.register_buffer("dc_cpu_cap", torch.as_tensor(model.dc_cpu_cap, dtype=torch.float32))
        self.register_buffer("dc_idle_power", torch.as_tensor(model.dc_idle_power, dtype=torch.float32))
        self.register_buffer("phi_interactive", torch.as_tensor(model.phi_interactive, dtype=torch.float32))
        self.alpha_interactive = float(model.alpha_interactive)
        self.alpha_batch = float(model.alpha_batch)

        valid = torch.zeros((self.J, self.T), dtype=torch.float32)
        for j in range(self.J):
            rel = int(model.release_times[j])
            ddl = int(model.deadlines[j])
            valid[j, rel : ddl + 1] = 1.0
        self.register_buffer("job_valid_mask", valid)

    def forward(self, x_red, renew_avail, interactive_demand):
        B = x_red.shape[0]
        sl = self.red_slices

        p_th = x_red[:, sl["p_th"]]
        u_th = x_red[:, sl["u_th"]]
        p_grid = x_red[:, sl["p_grid"]]
        ren_use_raw = x_red[:, sl["ren_use"]].reshape(B, self.D, self.T)
        y_4d = x_red[:, sl["y"]].reshape(B, self.D, self.T)
        y = y_4d.reshape(B, -1)

        xI_red = x_red[:, sl["xI_red"]].reshape(B, self.R, self.Dm1, self.T)
        z_red = x_red[:, sl["z_red"]].reshape(B, self.J, self.Dm1, self.T)

        xI_last = 1.0 - xI_red.sum(dim=2, keepdim=True)
        z_target = self.job_valid_mask.view(1, self.J, 1, self.T)
        z_last = z_target - z_red.sum(dim=2, keepdim=True)

        xI_raw = torch.cat([xI_red, xI_last], dim=2)
        # Project to simplex: xI >= 0 and sum_d xI = 1 for each (r,t).
        xI_pos = torch.relu(xI_raw) + 1.0e-6
        xI_full_4d = xI_pos / xI_pos.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        xI_full = xI_full_4d.reshape(B, -1)

        z_raw = torch.cat([z_red, z_last], dim=2)
        # Project to simplex with target sum (0/1): z >= 0 and sum_d z = target.
        z_pos = torch.relu(z_raw) + 1.0e-6
        z_norm = z_pos / z_pos.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        z_full_4d = z_target * z_norm
        z_full = z_full_4d.reshape(B, -1)

        f_raw = x_red[:, sl["f"]].reshape(B, self.J, self.D, self.T)
        f_cap = torch.relu(self.dc_cpu_cap.view(1, 1, self.D, 1) * z_full_4d)
        f_4d = torch.minimum(torch.relu(f_raw), f_cap)
        f = f_4d.reshape(B, -1)

        li = (
            self.phi_interactive.view(1, self.R, 1, 1)
            * interactive_demand.unsqueeze(2)
            * xI_full_4d
        ).sum(dim=1)
        batch_proc = f_4d.sum(dim=1)
        p_dc = (
            self.dc_idle_power.view(1, self.D, 1) * y_4d
            + self.alpha_interactive * li
            + self.alpha_batch * batch_proc
        )

        ren_upper = torch.minimum(torch.relu(renew_avail), torch.relu(p_dc))
        ren_use = torch.minimum(torch.relu(ren_use_raw), ren_upper).reshape(B, -1)
        delta = torch.ones((B, self.J), dtype=x_red.dtype, device=x_red.device)

        return torch.cat([p_th, u_th, p_grid, ren_use, y, xI_full, z_full, f, delta], dim=1)


def _build_reconstruct_node(model, red_slices):
    import neuromancer as nm

    reconstruct = _EqualityReconstructNode(model, red_slices)
    return nm.system.Node(
        reconstruct,
        ["x_red_rnd", "renew_avail", "interactive_demand"],
        ["x_rnd"],
        name="eq_reconstruct",
    )


def _first_n_samples(loader, n):
    datadict = loader.dataset.datadict
    keys = list(datadict.keys())
    out = []
    m = min(n, len(loader.dataset))
    for i in range(m):
        out.append({k: datadict[k][i] for k in keys})
    return out


def _multidc_constraint_labels(model):
    """Readable labels in the exact add-order of math_solver.multi_dc constraints."""
    labels = []

    T = int(model.horizon)
    D = int(model.num_dc)
    R = int(model.num_regions)
    J = int(model.num_jobs)
    E = int(model.num_branch)

    # Core power / generation constraints.
    for t in range(T):
        labels.append(f"thermal_cap[t={t}]")
        for d in range(D):
            labels.append(f"renew_avail_cap[d={d},t={t}]")
            labels.append(f"renew_local_use_cap[d={d},t={t}]")

    # Interactive assignment + latency feasibility.
    for r in range(R):
        for t in range(T):
            labels.append(f"interactive_assign_sum[r={r},t={t}]")
            for d in range(D):
                labels.append(f"interactive_latency[r={r},d={d},t={t}]")

    # Batch location and processing windows.
    for j in range(J):
        rel = int(model.release_times[j])
        ddl = int(model.deadlines[j])
        for t in range(T):
            if rel <= t <= ddl:
                labels.append(f"batch_location_active[j={j},t={t}]")
            else:
                labels.append(f"batch_location_inactive[j={j},t={t}]")
            for d in range(D):
                labels.append(f"batch_proc_cap[j={j},d={d},t={t}]")
        labels.append(f"batch_completion[j={j}]")

    # Activation consistency and per-DC capacity.
    for d in range(D):
        for t in range(T):
            labels.append(f"dc_capacity[d={d},t={t}]")
            for r in range(R):
                labels.append(f"interactive_requires_on[r={r},d={d},t={t}]")
            for j in range(J):
                labels.append(f"batch_requires_on[j={j},d={d},t={t}]")

    # DistFlow constraints.
    for t in range(T):
        labels.append(f"root_p_coupling[t={t}]")
        labels.append(f"root_q_coupling[t={t}]")
        labels.append(f"root_voltage_fix[t={t}]")

        for e in range(E):
            labels.append(f"distflow_pf_ub[e={e},t={t}]")
            labels.append(f"distflow_pf_lb[e={e},t={t}]")
            labels.append(f"distflow_qf_ub[e={e},t={t}]")
            labels.append(f"distflow_qf_lb[e={e},t={t}]")
            labels.append(f"distflow_vdrop[e={e},t={t}]")

        for n in range(1, int(model.num_bus)):
            pe_idx = int(model.parent_edge_of_bus[n])
            if pe_idx < 0:
                continue
            labels.append(f"node_p_balance[n={n},t={t}]")
            labels.append(f"node_q_balance[n={n},t={t}]")

        labels.append(f"root_p_agg[t={t}]")
        labels.append(f"root_q_agg[t={t}]")

    # Switching and migration linearizations.
    for d in range(D):
        labels.append(f"switch_init_pos[d={d}]")
        labels.append(f"switch_init_neg[d={d}]")
        for t in range(1, T):
            labels.append(f"switch_pos[d={d},t={t}]")
            labels.append(f"switch_neg[d={d},t={t}]")

    for t in range(1, T):
        for r in range(R):
            for d in range(D):
                for dp in range(D):
                    if d == dp:
                        continue
                    labels.append(f"mig_interactive[r={r},d={d},dp={dp},t={t}]")
        for j in range(J):
            for d in range(D):
                for dp in range(D):
                    if d == dp:
                        continue
                    labels.append(f"mig_batch[j={j},d={d},dp={dp},t={t}]")

    return labels


def _print_most_violated(title: str, labels, counts, topk: int = 5):
    if len(counts) == 0:
        return

    top_idx = int(np.argmax(counts))
    print(f"=== Most Violated Constraint ({title}) ===")
    print(f"{labels[top_idx]} -> {int(counts[top_idx])} cases")

    k = min(int(topk), len(counts))
    order = np.argsort(-counts)[:k]
    print(f"Top-{k} {title} violation counts:")
    for idx in order:
        idx = int(idx)
        print(f"  {labels[idx]}: {int(counts[idx])}")


def _save_constraint_stats(
    labels,
    policy_counts,
    n_cases: int,
    method,
    penalty,
    horizon,
    suffix: str,
):
    if len(labels) == 0:
        return None

    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    stats = pd.DataFrame(
        {
            "Constraint": labels,
            "Policy Violation Count": np.asarray(policy_counts, dtype=int),
            "Policy Violation Rate": np.asarray(policy_counts, dtype=float) / max(int(n_cases), 1),
        }
    )

    stats = stats.sort_values("Policy Violation Count", ascending=False)
    out_path = result_dir / f"mdc_constraint_stats_{method}{penalty}_T{horizon}{suffix}.csv"
    stats.to_csv(out_path, index=False)
    print(f"[ConstraintStats] Saved: {out_path}")
    return out_path


def _save_violation_count_table(constraint_labels, violation_counts, out_path):
    """Save a compact two-column table: constraint name and violation count."""
    tbl = pd.DataFrame(
        {
            "Constraint": list(constraint_labels),
            "Violation Count": np.asarray(violation_counts, dtype=int),
        }
    ).sort_values("Violation Count", ascending=False)
    tbl.to_csv(out_path, index=False)
    return out_path


def _effective_timelimit(config, default: float = 60.0):
    """Return None when solver_time_limit<=0, meaning no solver time limit."""
    raw = getattr(config, "solver_time_limit", default)
    try:
        val = float(raw)
    except Exception:
        val = float(default)
    return None if val <= 0.0 else val


def _extract_solver_meta(model):
    """Best-effort extraction of solver termination metadata from Pyomo results."""
    out = {
        "solver_status": "na",
        "solver_termination": "na",
        "solver_mip_gap": np.nan,
        "solver_lower_bound": np.nan,
        "solver_upper_bound": np.nan,
    }

    res = getattr(model, "res", None)
    if res is None:
        return out

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return np.nan

    try:
        solver_info = getattr(res, "solver", None)
        if solver_info is not None:
            st = getattr(solver_info, "status", None)
            tc = getattr(solver_info, "termination_condition", None)
            if st is not None:
                out["solver_status"] = str(st)
            if tc is not None:
                out["solver_termination"] = str(tc)

            for key in ["mip_gap", "gap", "relative_gap", "MIPGap", "Gap"]:
                val = getattr(solver_info, key, None)
                if val is not None:
                    gap_val = _to_float(val)
                    if np.isfinite(gap_val):
                        out["solver_mip_gap"] = gap_val
                        break
    except Exception:
        pass

    # Try problem bounds from the first problem entry if available.
    try:
        problem_info = getattr(res, "problem", None)
        first_prob = None
        if problem_info is not None:
            if hasattr(problem_info, "values"):
                vals = list(problem_info.values())
                if len(vals) > 0:
                    first_prob = vals[0]
            if first_prob is None:
                first_prob = problem_info

        if first_prob is not None:
            lb = getattr(first_prob, "lower_bound", None)
            ub = getattr(first_prob, "upper_bound", None)
            out["solver_lower_bound"] = _to_float(lb)
            out["solver_upper_bound"] = _to_float(ub)

            # If solver didn't expose relative gap directly, compute a stable proxy.
            if not np.isfinite(out["solver_mip_gap"]):
                lbv = out["solver_lower_bound"]
                ubv = out["solver_upper_bound"]
                if np.isfinite(lbv) and np.isfinite(ubv):
                    out["solver_mip_gap"] = abs(ubv - lbv) / max(abs(ubv), 1.0e-12)
    except Exception:
        pass

    return out


def _hard_binarize_policy_x(model, x_policy: np.ndarray, threshold: float = 0.5):
    """Project policy vector to strict 0/1 on binary slices for consistent Pyomo-side evaluation."""
    x = np.asarray(x_policy, dtype=float).reshape(-1).copy()
    if x.size < int(model.nx):
        raise ValueError(f"policy output dim={x.size} is smaller than model.nx={model.nx}")

    # Binary on/off style variables keep thresholding.
    for key in ["u_th", "y"]:
        if key in model.x_slices:
            sl = model.x_slices[key]
            x[sl] = (x[sl] >= float(threshold)).astype(float)

    # Assignment variables should be discretized with argmax over DC dimension,
    # not elementwise threshold, otherwise dense soft assignments can collapse to all-zero.
    T = int(model.horizon)
    D = int(model.num_dc)
    R = int(model.num_regions)
    J = int(model.num_jobs)

    if "xI" in model.x_slices:
        sl = model.x_slices["xI"]
        xi = x[sl].reshape(R, D, T)
        xi_oh = np.zeros_like(xi)
        best = np.argmax(xi, axis=1)
        for r in range(R):
            for t in range(T):
                xi_oh[r, best[r, t], t] = 1.0
        x[sl] = xi_oh.reshape(-1)

    if "z" in model.x_slices:
        sl = model.x_slices["z"]
        z = x[sl].reshape(J, D, T)
        z_oh = np.zeros_like(z)
        best = np.argmax(z, axis=1)
        for j in range(J):
            rel = int(model.release_times[j]) if hasattr(model, "release_times") else 0
            ddl = int(model.deadlines[j]) if hasattr(model, "deadlines") else (T - 1)
            rel = max(0, rel)
            ddl = min(T - 1, ddl)
            for t in range(T):
                if rel <= t <= ddl:
                    z_oh[j, best[j, t], t] = 1.0
        x[sl] = z_oh.reshape(-1)

    # Hard-constraint regime: every job must be completed.
    if "delta" in model.x_slices:
        sl = model.x_slices["delta"]
        x[sl] = 1.0

    return x


def _assign_policy_solution_for_violation(model, x_policy: np.ndarray):
    """Write policy output into Pyomo vars so per-constraint violation can be evaluated."""
    x = np.asarray(x_policy, dtype=float).reshape(-1)
    if x.size < int(model.nx):
        raise ValueError(f"policy output dim={x.size} is smaller than model.nx={model.nx}")

    # Initialize all variables to zero to keep constraint body evaluable.
    for _, var_comp in model.vars.items():
        for idx in var_comp:
            var_comp[idx].set_value(0.0)

    T = int(model.horizon)
    D = int(model.num_dc)
    R = int(model.num_regions)
    J = int(model.num_jobs)

    sl = model.x_slices

    p_th = x[sl["p_th"]]
    u_th = x[sl["u_th"]]
    p_grid = x[sl["p_grid"]]
    ren_use = x[sl["ren_use"]].reshape(D, T)
    y = x[sl["y"]].reshape(D, T)
    xI = x[sl["xI"]].reshape(R, D, T)
    z = x[sl["z"]].reshape(J, D, T)
    f = x[sl["f"]].reshape(J, D, T)
    delta = x[sl["delta"]]

    for t in range(T):
        model.vars["p_th"][t].set_value(float(p_th[t]))
        model.vars["u_th"][t].set_value(float(u_th[t]))
        model.vars["p_grid"][t].set_value(float(p_grid[t]))
        for d in range(D):
            model.vars["ren_use"][d, t].set_value(float(ren_use[d, t]))
            model.vars["y"][d, t].set_value(float(y[d, t]))

    for r in range(R):
        for d in range(D):
            for t in range(T):
                model.vars["xI"][r, d, t].set_value(float(xI[r, d, t]))

    for j in range(J):
        model.vars["delta"][j].set_value(float(delta[j]))
        for d in range(D):
            for t in range(T):
                model.vars["z"][j, d, t].set_value(float(z[j, d, t]))
                model.vars["f"][j, d, t].set_value(float(f[j, d, t]))

    # Map DC-level on/off y to node-level on/off binaries for node-switch constraints.
    if "on" in model.vars and hasattr(model, "num_nodes_per_dc"):
        for d in range(D):
            n_nodes = int(model.num_nodes_per_dc[d])
            for t in range(T):
                ydt = float(np.clip(y[d, t], 0.0, 1.0))
                n_on = int(np.round(ydt * n_nodes))
                n_on = max(0, min(n_on, n_nodes))
                for k in range(n_nodes):
                    model.vars["on"][d, k, t].set_value(1.0 if k < n_on else 0.0)

    if "nu" in model.vars and "on" in model.vars and hasattr(model, "num_nodes_per_dc"):
        for d in range(D):
            n_nodes = int(model.num_nodes_per_dc[d])
            for k in range(n_nodes):
                init_k = float(model.node_init[d, k]) if hasattr(model, "node_init") else float(model.y_init[d] >= 0.5)
                on0 = float(model.vars["on"][d, k, 0].value)
                model.vars["nu"][d, k, 0].set_value(abs(on0 - init_k))
                for t in range(1, T):
                    on_t = float(model.vars["on"][d, k, t].value)
                    on_prev = float(model.vars["on"][d, k, t - 1].value)
                    model.vars["nu"][d, k, t].set_value(abs(on_t - on_prev))

    # Reconstruct DistFlow state from assigned scheduling variables so node/root
    # power balance constraints are evaluated on a physically consistent point.
    base_load = np.array([float(model.params["base_load"][t].value) for t in range(T)], dtype=float)
    interactive = np.array(
        [[float(model.params["interactive_demand"][r, t].value) for t in range(T)] for r in range(R)],
        dtype=float,
    )

    p_dc = np.zeros((D, T), dtype=float)
    for d in range(D):
        for t in range(T):
            li = 0.0
            for r in range(R):
                li += float(model.phi_interactive[r]) * interactive[r, t] * xI[r, d, t]
            batch_proc = float(np.sum(f[:, d, t]))
            p_dc[d, t] = (
                float(model.dc_idle_power[d]) * y[d, t]
                + float(model.alpha_interactive) * li
                + float(model.alpha_batch) * batch_proc
            )

    N = int(model.num_bus)
    E = int(model.num_branch)
    p_load = np.zeros((N, T), dtype=float)
    q_load = np.zeros((N, T), dtype=float)

    for n in range(N):
        p_load[n, :] = float(model.p_base_share[n]) * base_load
        q_load[n, :] = float(model.q_base_share[n]) * base_load

    for d in range(D):
        bus_idx = int(model.dc_bus_map[d])
        p_load[bus_idx, :] += p_dc[d, :] - ren_use[d, :]
        q_load[bus_idx, :] += float(model.dc_reactive_factor) * p_dc[d, :]

    p_subtree = p_load.copy()
    q_subtree = q_load.copy()

    # Build strict postorder traversal on the radial tree for exact subtree sums.
    postorder = []

    def _dfs_post(u: int):
        for e in model.children_edges_of_bus[u]:
            child = int(model.branch_to[int(e)])
            _dfs_post(child)
        postorder.append(u)

    _dfs_post(0)
    for n in postorder:
        if n == 0:
            continue
        e = int(model.parent_edge_of_bus[n])
        if e < 0:
            continue
        parent = int(model.branch_from[e])
        p_subtree[parent, :] += p_subtree[n, :]
        q_subtree[parent, :] += q_subtree[n, :]

    pf = np.zeros((E, T), dtype=float)
    qf = np.zeros((E, T), dtype=float)
    for e in range(E):
        child = int(model.branch_to[e])
        pf[e, :] = p_subtree[child, :]
        qf[e, :] = q_subtree[child, :]
        for t in range(T):
            model.vars["pf"][e, t].set_value(float(pf[e, t]))
            model.vars["qf"][e, t].set_value(float(qf[e, t]))

    # Voltage recursion on the radial feeder.
    v = np.zeros((N, T), dtype=float)
    v[0, :] = 1.0
    for n in range(1, N):
        e = int(model.parent_edge_of_bus[n])
        if e < 0:
            continue
        parent = int(model.branch_from[e])
        v[n, :] = v[parent, :] - 2.0 * (
            float(model.branch_r[e]) * pf[e, :] + float(model.branch_x[e]) * qf[e, :]
        )
    for n in range(N):
        for t in range(T):
            model.vars["v"][n, t].set_value(float(v[n, t]))

    root_children = model.children_edges_of_bus[0]
    p_sub = np.zeros(T, dtype=float)
    q_sub = np.zeros(T, dtype=float)
    for e in root_children:
        p_sub += pf[int(e), :]
        q_sub += qf[int(e), :]

    for t in range(T):
        model.vars["p_sub"][t].set_value(float(p_sub[t]))
        model.vars["q_sub"][t].set_value(float(q_sub[t]))
        model.vars["q_grid"][t].set_value(float(q_sub[t]))
        # Enforce root active power coupling equality exactly in the violation check point.
        model.vars["p_grid"][t].set_value(float(p_sub[t] - p_th[t]))

    # Auxiliary variables for switching / migration linearization constraints.
    for d in range(D):
        model.vars["eta"][d, 0].set_value(float(abs(y[d, 0] - float(model.y_init[d]))))
        for t in range(1, T):
            model.vars["eta"][d, t].set_value(float(abs(y[d, t] - y[d, t - 1])))

    for r in range(R):
        for d in range(D):
            for dp in range(D):
                if d == dp:
                    continue
                for t in range(1, T):
                    val = max(0.0, xI[r, d, t - 1] + xI[r, dp, t] - 1.0)
                    model.vars["mI"][r, d, dp, t].set_value(float(val))

    for j in range(J):
        for d in range(D):
            for dp in range(D):
                if d == dp:
                    continue
                for t in range(1, T):
                    val = max(0.0, z[j, d, t - 1] + z[j, dp, t] - 1.0)
                    model.vars["mB"][j, d, dp, t].set_value(float(val))


def _check_integrality(model, tol: float = 1e-6):
    """Check whether binary/integer variables are integral after policy assignment."""
    n_bin = 0
    n_int = 0
    n_bin_nonint = 0
    n_int_nonint = 0
    max_bin_dev = 0.0
    max_int_dev = 0.0

    for _, var_comp in model.vars.items():
        for idx in var_comp:
            v = var_comp[idx]
            try:
                val = float(pe.value(v))
            except Exception:
                continue
            if not np.isfinite(val):
                continue

            if bool(v.is_binary()):
                n_bin += 1
                dev_round = abs(val - float(np.round(val)))
                dev_01 = min(abs(val - 0.0), abs(val - 1.0))
                dev = max(dev_round, dev_01)
                max_bin_dev = max(max_bin_dev, dev)
                if dev > tol:
                    n_bin_nonint += 1
            elif bool(v.is_integer()):
                n_int += 1
                dev = abs(val - float(np.round(val)))
                max_int_dev = max(max_int_dev, dev)
                if dev > tol:
                    n_int_nonint += 1

    return {
        "n_bin": int(n_bin),
        "n_int": int(n_int),
        "n_bin_nonint": int(n_bin_nonint),
        "n_int_nonint": int(n_int_nonint),
        "max_bin_dev": float(max_bin_dev),
        "max_int_dev": float(max_int_dev),
    }


def _collect_policy_schedule_rows(model, sample_id: int):
    """Flatten all decision variable values into row records for CSV export."""
    rows = []
    for var_name, var_comp in model.vars.items():
        for idx in var_comp:
            try:
                val = float(pe.value(var_comp[idx]))
            except Exception:
                val = np.nan

            if idx is None:
                idx_str = ""
            elif isinstance(idx, tuple):
                idx_str = "|".join(str(v) for v in idx)
            else:
                idx_str = str(idx)

            rows.append(
                {
                    "sample_id": int(sample_id),
                    "var_name": str(var_name),
                    "index": idx_str,
                    "value": val,
                }
            )
    return rows


def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)

    seed = int(getattr(config, "seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    import neuromancer as nm
    from src.func import roundGumbelModel
    from src.func.layer import netFC
    from src.problem.math_solver.multi_dc import multiDC as msMultiDC
    from src.problem.neuromancer.multi_dc import penaltyLoss as nmMultiDCLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = int(config.horizon)
    D = int(config.num_dc)
    R = int(config.num_regions)
    J = int(config.num_jobs)

    model_kwargs = {}
    for key in ["dc_cpu_cap", "dc_bus_map", "num_nodes_per_dc", "cpu_per_node"]:
        if hasattr(config, key) and getattr(config, key) is not None:
            model_kwargs[key] = getattr(config, key)

    model = msMultiDC(
        horizon=T,
        num_dc=D,
        num_regions=R,
        num_jobs=J,
        solver=getattr(config, "solver", "highs"),
        timelimit=_effective_timelimit(config, default=60.0),
        **model_kwargs,
    )

    param_keys = ["base_load", "price", "renew_avail_flat", "interactive_flat", "batch_work"]
    in_dim = _flatten_param_dim(T, D, R, J)
    red_slices, nx_red, int_ind_red, bin_ind_red = _build_reduced_x_layout(model)

    concat_node = _build_param_concat_node(param_keys)
    smap_func = _build_smap(config, in_dim=in_dim, out_dim=nx_red)
    smap = nm.system.Node(smap_func, ["xi"], ["x_red"], name="smap")

    hsize = int(getattr(config, "hsize", 128))
    hlayers_rnd = int(getattr(config, "hlayers_rnd", 4))
    rnd_layers = netFC(input_dim=in_dim + nx_red, hidden_dims=[hsize] * hlayers_rnd, output_dim=nx_red)

    rnd = roundGumbelModel(
        layers=rnd_layers,
        param_keys=["xi"],
        var_keys=["x_red"],
        output_keys=["x_red_rnd"],
        int_ind=int_ind_red,
        bin_ind=bin_ind_red,
        continuous_update=True,
        name="round",
    )
    reconstruct_node = _build_reconstruct_node(model, red_slices)

    components = nn.ModuleList([concat_node, smap, rnd, reconstruct_node]).to(device)

    loss_fn = nmMultiDCLoss(
        ["base_load", "price", "renew_avail", "interactive_demand", "batch_work", "x_rnd"],
        horizon=T,
        num_dc=D,
        num_regions=R,
        num_jobs=J,
        x_slices=model.x_slices,
        latency_mask=model.latency_mask,
        release_times=model.release_times,
        deadlines=model.deadlines,
        p_th_max=model.p_th_max,
        p_grid_max=model.p_grid_max,
        dc_idle_power=model.dc_idle_power,
        dc_cpu_cap=model.dc_cpu_cap,
        phi_interactive=model.phi_interactive,
        alpha_interactive=model.alpha_interactive,
        alpha_batch=model.alpha_batch,
        thermal_quad=model.thermal_quad,
        thermal_lin=model.thermal_lin,
        thermal_on_cost=model.thermal_on_cost,
        switch_cost=model.switch_cost,
        num_nodes_per_dc=model.num_nodes_per_dc,
        mig_cost_interactive=model.mig_cost_interactive,
        mig_cost_batch=model.mig_cost_batch,
        curtail_cost=model.curtail_cost,
        late_penalty=model.late_penalty,
        y_init=model.y_init,
        p_base_share=model.p_base_share,
        q_base_share=model.q_base_share,
        branch_from=model.branch_from,
        branch_to=model.branch_to,
        branch_r=model.branch_r,
        branch_x=model.branch_x,
        parent_edge_of_bus=model.parent_edge_of_bus,
        children_edges_of_bus=model.children_edges_of_bus,
        dc_bus_map=model.dc_bus_map,
        line_flow_abs_max=model.line_flow_abs_max,
        v_min_sq=model.v_min_sq,
        v_max_sq=model.v_max_sq,
        dc_reactive_factor=model.dc_reactive_factor,
        penalty_weight=float(getattr(config, "penalty", 40.0)),
        eq_weight=float(getattr(config, "eq_weight", 2.0)),
        obj_weight=float(getattr(config, "obj_weight", 1.0)),
        viol_weight=float(getattr(config, "viol_weight", 1.0)),
    ).to(device)

    utils.train(
        components,
        loss_fn,
        loader_train,
        loader_val,
        lr=float(getattr(config, "lr", 1e-3)),
        penalty_growth=penalty_growth,
        patience=int(getattr(config, "patience", 20)),
        warmup=getattr(config, "warmup", None),
        validate_every=int(getattr(config, "validate_every", 100)),
        train_eval_batches=int(getattr(config, "train_eval_batches", 8)),
        loader_test=loader_test,
        tensorboard=bool(getattr(config, "tb", False)),
        tb_logdir=getattr(config, "tb_logdir", "runs"),
        tb_run_name=f"multidc_cls_T{T}_pen{getattr(config, 'penalty', 40.0)}",
        epochs=int(getattr(config, "epochs", 200)),
    )

    df = evaluate(components, loss_fn, model, loader_test, config)

    suffix = "-g" if penalty_growth else ""
    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"mdc_cls{getattr(config, 'penalty', 40.0)}_T{T}{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"[Multi-DC] Saved result: {out_path}")


def exact(loader_test, config):
    from src.problem.math_solver.multi_dc import multiDC as msMultiDC

    T = int(config.horizon)
    D = int(config.num_dc)
    R = int(config.num_regions)
    J = int(config.num_jobs)

    model_kwargs = {}
    for key in ["dc_cpu_cap", "dc_bus_map", "num_nodes_per_dc", "cpu_per_node"]:
        if hasattr(config, key) and getattr(config, key) is not None:
            model_kwargs[key] = getattr(config, key)

    model = msMultiDC(
        horizon=T,
        num_dc=D,
        num_regions=R,
        num_jobs=J,
        solver=getattr(config, "solver", "highs"),
        timelimit=_effective_timelimit(config, default=60.0),
        **model_kwargs,
    )

    n_eval = int(getattr(config, "eval_samples", 50))
    samples = _first_n_samples(loader_test, n_eval)

    constraint_labels = None
    solver_viol_count = None

    rows = []
    for dp in tqdm(samples, desc="Solver-Only"):
        params = {
            "base_load": dp["base_load"].cpu().numpy(),
            "price": dp["price"].cpu().numpy(),
            "renew_avail": dp["renew_avail"].cpu().numpy(),
            "interactive_demand": dp["interactive_demand"].cpu().numpy(),
            "batch_work": dp["batch_work"].cpu().numpy(),
        }
        tick = time.time()
        try:
            model.set_param_val(params)
            _, obj = model.solve(tee=bool(getattr(config, "solver_tee", False)))
            viol = np.asarray(model.cal_violation(), dtype=float)
            solver_meta = _extract_solver_meta(model)

            if constraint_labels is None:
                constraint_labels = _multidc_constraint_labels(model)
                if len(constraint_labels) != len(viol):
                    constraint_labels = [f"cons_{k + 1}" for k in range(len(viol))]
                solver_viol_count = np.zeros(len(viol), dtype=int)
            if len(viol) == len(solver_viol_count):
                solver_viol_count += (viol > 1e-6).astype(int)

            tock = time.time()
            rows.append(
                {
                    "solver_obj": float(obj),
                    "solver_mean_viol": float(np.mean(viol)),
                    "solver_max_viol": float(np.max(viol)),
                    "solver_num_viol": int(np.sum(viol > 1e-6)),
                    "solver_time": float(tock - tick),
                    "solver_status": solver_meta["solver_status"],
                    "solver_termination": solver_meta["solver_termination"],
                    "solver_mip_gap": solver_meta["solver_mip_gap"],
                    "solver_lower_bound": solver_meta["solver_lower_bound"],
                    "solver_upper_bound": solver_meta["solver_upper_bound"],
                    "status": "ok",
                }
            )
        except Exception as e:
            tock = time.time()
            rows.append(
                {
                    "solver_obj": np.nan,
                    "solver_mean_viol": np.nan,
                    "solver_max_viol": np.nan,
                    "solver_num_viol": np.nan,
                    "solver_time": float(tock - tick),
                    "solver_status": "na",
                    "solver_termination": "na",
                    "solver_mip_gap": np.nan,
                    "solver_lower_bound": np.nan,
                    "solver_upper_bound": np.nan,
                    "status": f"fail:{type(e).__name__}",
                }
            )

    df = pd.DataFrame(rows)
    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    # Keep per-sample solver metrics in a separate file.
    sample_out_path = result_dir / f"mdc_solver_samples_T{T}.csv"
    df.to_csv(sample_out_path, index=False)
    print(f"[Multi-DC] Saved solver sample metrics: {sample_out_path}")
    print(df.describe(include="all"))

    if solver_viol_count is not None and len(solver_viol_count) > 0:
        _print_most_violated("Solver", constraint_labels, solver_viol_count, topk=5)

        # Save the main solver CSV as requested: two columns only.
        out_path = result_dir / f"mdc_solver_T{T}.csv"
        _save_violation_count_table(constraint_labels, solver_viol_count, out_path)
        print(f"[Multi-DC] Saved solver violation table: {out_path}")

        _save_constraint_stats(
            labels=constraint_labels,
            policy_counts=solver_viol_count,
            n_cases=len(df),
            method="solver",
            penalty="na",
            horizon=T,
            suffix="",
        )


def evaluate(components, loss_fn, model, loader_test, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    components.eval()

    n_eval = int(getattr(config, "eval_samples", 50))
    # Follow CLI/runtime switch: policy-only runs can fully skip solver evaluation.
    compare_solver = bool(getattr(config, "compare_solver", False))

    constraint_labels = None
    policy_viol_count = None
    policy_viol_eval_errors = 0
    integrality_eval_errors = 0
    policy_schedule_rows = []

    samples = _first_n_samples(loader_test, n_eval)
    rows = []

    for sample_id, dp in enumerate(tqdm(samples, desc="Policy-vs-Solver" if compare_solver else "Policy")):
        xdp = {
            "base_load": dp["base_load"].unsqueeze(0).to(device),
            "price": dp["price"].unsqueeze(0).to(device),
            "renew_avail": dp["renew_avail"].unsqueeze(0).to(device),
            "interactive_demand": dp["interactive_demand"].unsqueeze(0).to(device),
            "batch_work": dp["batch_work"].unsqueeze(0).to(device),
            "renew_avail_flat": dp["renew_avail_flat"].unsqueeze(0).to(device),
            "interactive_flat": dp["interactive_flat"].unsqueeze(0).to(device),
        }

        policy_tick = time.time()
        with torch.no_grad():
            for comp in components:
                xdp.update(comp(xdp))
            xdp = loss_fn(xdp)

            pol_obj_nm = float(loss_fn.cal_obj(xdp).detach().cpu().numpy().reshape(-1)[0])
            pol_viol_nm = float(loss_fn.cal_constr_viol(xdp).detach().cpu().numpy().reshape(-1)[0])
        policy_tock = time.time()

        x_pol_raw = xdp["x_rnd"].detach().cpu().numpy().reshape(-1)
        x_pol_hard = _hard_binarize_policy_x(model, x_pol_raw)
        hard_flip_ratio = float(np.mean(np.abs(x_pol_hard - x_pol_raw) > 1e-8))

        row = {
            # Keep neuromancer-side metrics for debugging/ablation.
            "policy_obj_nm": pol_obj_nm,
            "policy_violation_nm": pol_viol_nm,
            # Keep track of how much hard projection changed policy values.
            "policy_hard_flip_ratio": hard_flip_ratio,
            "policy_time": float(policy_tock - policy_tick),
        }

        # Compute policy-side per-constraint violations via Pyomo model for detailed stats.
        params = {
            "base_load": dp["base_load"].cpu().numpy(),
            "price": dp["price"].cpu().numpy(),
            "renew_avail": dp["renew_avail"].cpu().numpy(),
            "interactive_demand": dp["interactive_demand"].cpu().numpy(),
            "batch_work": dp["batch_work"].cpu().numpy(),
        }
        try:
            model.set_param_val(params)
            _assign_policy_solution_for_violation(model, x_pol_hard)
            policy_schedule_rows.extend(_collect_policy_schedule_rows(model, sample_id=sample_id))
            integ = _check_integrality(model)
            pol_obj_pyomo = float(pe.value(model.model.obj))
            pol_viol_arr = np.asarray(model.cal_violation(), dtype=float)

            row.update(
                {
                    # Use Pyomo-side objective/violation for strict policy-vs-solver comparison.
                    "policy_obj": pol_obj_pyomo,
                    "policy_mean_viol": float(np.mean(pol_viol_arr)),
                    "policy_max_viol": float(np.max(pol_viol_arr)),
                    "policy_num_viol": int(np.sum(pol_viol_arr > 1e-6)),
                    "policy_n_binary": int(integ["n_bin"]),
                    "policy_n_integer": int(integ["n_int"]),
                    "policy_binary_nonint": int(integ["n_bin_nonint"]),
                    "policy_integer_nonint": int(integ["n_int_nonint"]),
                    "policy_binary_max_dev": float(integ["max_bin_dev"]),
                    "policy_integer_max_dev": float(integ["max_int_dev"]),
                }
            )

            if constraint_labels is None:
                constraint_labels = _multidc_constraint_labels(model)
                if len(constraint_labels) != len(pol_viol_arr):
                    constraint_labels = [f"cons_{k + 1}" for k in range(len(pol_viol_arr))]
                policy_viol_count = np.zeros(len(pol_viol_arr), dtype=int)

            if len(pol_viol_arr) == len(policy_viol_count):
                policy_viol_count += (pol_viol_arr > 1e-6).astype(int)
        except Exception:
            # Keep evaluation robust: stats are best-effort and should not block result generation.
            policy_viol_eval_errors += 1
            row.update(
                {
                    "policy_obj": pol_obj_nm,
                    "policy_mean_viol": np.nan,
                    "policy_max_viol": np.nan,
                    "policy_num_viol": np.nan,
                    "policy_n_binary": np.nan,
                    "policy_n_integer": np.nan,
                    "policy_binary_nonint": np.nan,
                    "policy_integer_nonint": np.nan,
                    "policy_binary_max_dev": np.nan,
                    "policy_integer_max_dev": np.nan,
                }
            )
            integrality_eval_errors += 1

        if compare_solver:
            tick = time.time()
            try:
                model.set_param_val(params)
                _, obj = model.solve(tee=bool(getattr(config, "solver_tee", False)))
                viol = np.asarray(model.cal_violation(), dtype=float)
                solver_meta = _extract_solver_meta(model)

                tock = time.time()
                row.update(
                    {
                        "solver_obj": float(obj),
                        "solver_mean_viol": float(np.mean(viol)),
                        "solver_max_viol": float(np.max(viol)),
                        "solver_num_viol": int(np.sum(viol > 1e-6)),
                        "solver_time": float(tock - tick),
                        "solver_status": solver_meta["solver_status"],
                        "solver_termination": solver_meta["solver_termination"],
                        "solver_mip_gap": solver_meta["solver_mip_gap"],
                        "solver_lower_bound": solver_meta["solver_lower_bound"],
                        "solver_upper_bound": solver_meta["solver_upper_bound"],
                        "obj_gap_policy_minus_solver": float(float(row["policy_obj"]) - float(obj)),
                        "status": "ok",
                    }
                )
            except Exception as e:
                tock = time.time()
                row.update(
                    {
                        "solver_obj": np.nan,
                        "solver_mean_viol": np.nan,
                        "solver_max_viol": np.nan,
                        "solver_num_viol": np.nan,
                        "solver_time": float(tock - tick),
                        "solver_status": "na",
                        "solver_termination": "na",
                        "solver_mip_gap": np.nan,
                        "solver_lower_bound": np.nan,
                        "solver_upper_bound": np.nan,
                        "obj_gap_policy_minus_solver": np.nan,
                        "status": f"fail:{type(e).__name__}",
                    }
                )

        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.describe(include="all"))

    # Reliability mask for solver baseline in minimization comparison.
    # Treat only proven-optimal runs as strict baseline; timed-out incumbents are not final optima.
    optimal_terms = {"optimal", "globallyOptimal", "locallyOptimal"}
    if "solver_termination" in df.columns:
        term_series = df["solver_termination"].astype(str)
        is_opt = term_series.isin(optimal_terms)
    else:
        is_opt = pd.Series([False] * len(df), index=df.index)

    if "obj_gap_policy_minus_solver" in df.columns:
        df["obj_gap_policy_minus_solver_reliable"] = np.where(
            is_opt,
            pd.to_numeric(df["obj_gap_policy_minus_solver"], errors="coerce"),
            np.nan,
        )

    if "obj_gap_policy_minus_solver" in df.columns:
        gap_series = df["obj_gap_policy_minus_solver"].dropna()
        if len(gap_series) > 0:
            pol_mean = float(pd.to_numeric(df["policy_obj"], errors="coerce").dropna().mean())
            sol_mean = float(pd.to_numeric(df["solver_obj"], errors="coerce").dropna().mean())
            gap_mean = float(gap_series.mean())
            gap_abs_mean = float(gap_series.abs().mean())
            print("=== Final Test Objective Gap (Learning vs Solver) ===")
            print(f"policy_obj_mean={pol_mean:.6f}")
            print(f"solver_obj_mean={sol_mean:.6f}")
            print(f"gap_mean(policy-solver)={gap_mean:.6f}")
            print(f"gap_abs_mean={gap_abs_mean:.6f}")

    if "obj_gap_policy_minus_solver_reliable" in df.columns:
        reliable_gap = pd.to_numeric(df["obj_gap_policy_minus_solver_reliable"], errors="coerce").dropna()
        print("=== Reliable Objective Gap (Optimal Solver Only) ===")
        print(f"optimal_solver_cases={int(reliable_gap.shape[0])}")
        if len(reliable_gap) > 0:
            print(f"reliable_gap_mean(policy-solver)={float(reliable_gap.mean()):.6f}")
            print(f"reliable_gap_abs_mean={float(reliable_gap.abs().mean()):.6f}")
        else:
            print("reliable_gap_mean(policy-solver)=NaN")
            print("reliable_gap_abs_mean=NaN")
            print("[Warning] No optimal solver samples; current gap vs solver incumbent is not a strict optimality comparison.")

    if "policy_time" in df.columns and "solver_time" in df.columns:
        pol_time_series = pd.to_numeric(df["policy_time"], errors="coerce").dropna()
        sol_time_series = pd.to_numeric(df["solver_time"], errors="coerce").dropna()
        if len(pol_time_series) > 0 and len(sol_time_series) > 0:
            pol_t_mean = float(pol_time_series.mean())
            sol_t_mean = float(sol_time_series.mean())
            speedup = (sol_t_mean / max(pol_t_mean, 1.0e-12))
            print("=== Solve Time Comparison (Learning vs Solver) ===")
            print(f"policy_time_mean_sec={pol_t_mean:.6f}")
            print(f"solver_time_mean_sec={sol_t_mean:.6f}")
            print(f"speedup_mean(solver/policy)={speedup:.2f}x")

    if "solver_termination" in df.columns:
        term_counts = df["solver_termination"].value_counts(dropna=False)
        print("=== Solver Termination Summary ===")
        for term, cnt in term_counts.items():
            print(f"{term}: {int(cnt)}")
        if "solver_mip_gap" in df.columns:
            gap_vals = pd.to_numeric(df["solver_mip_gap"], errors="coerce").dropna()
            if len(gap_vals) > 0:
                print(f"solver_mip_gap_mean={float(gap_vals.mean()):.6f}")

    if "policy_binary_nonint" in df.columns:
        b_nonint = pd.to_numeric(df["policy_binary_nonint"], errors="coerce")
        i_nonint = pd.to_numeric(df["policy_integer_nonint"], errors="coerce")
        b_max_dev = pd.to_numeric(df["policy_binary_max_dev"], errors="coerce")
        i_max_dev = pd.to_numeric(df["policy_integer_max_dev"], errors="coerce")
        n_bin = pd.to_numeric(df["policy_n_binary"], errors="coerce")
        n_int = pd.to_numeric(df["policy_n_integer"], errors="coerce")

        print("=== Policy Integrality Check ===")
        if n_bin.notna().any():
            print(f"binary_vars_per_sample~{int(n_bin.dropna().median()):d}")
        if n_int.notna().any():
            print(f"integer_vars_per_sample~{int(n_int.dropna().median()):d}")
        if b_nonint.notna().any():
            print(f"samples_with_noninteger_binary={int((b_nonint > 0).sum())}/{len(df)}")
        if i_nonint.notna().any():
            print(f"samples_with_noninteger_integer={int((i_nonint > 0).sum())}/{len(df)}")
        if b_max_dev.notna().any():
            print(f"max_binary_integrality_deviation={float(b_max_dev.max()):.3e}")
        if i_max_dev.notna().any():
            print(f"max_integer_integrality_deviation={float(i_max_dev.max()):.3e}")

    if policy_viol_eval_errors > 0:
        print(
            f"[PolicyViolationStats] skipped {policy_viol_eval_errors}/{len(samples)} samples "
            "due to evaluation errors"
        )

    if integrality_eval_errors > 0:
        print(
            f"[IntegralityCheck] skipped {integrality_eval_errors}/{len(samples)} samples "
            "due to evaluation errors"
        )

    if policy_viol_count is not None and len(policy_viol_count) > 0:
        _print_most_violated("Policy", constraint_labels, policy_viol_count, topk=5)

    if policy_viol_count is not None and len(policy_viol_count) > 0:
        suffix = "-g" if bool(getattr(config, "penalty_growth", False)) else ""
        _save_constraint_stats(
            labels=constraint_labels,
            policy_counts=policy_viol_count,
            n_cases=len(df),
            method=getattr(config, "method", "cls"),
            penalty=getattr(config, "penalty", "na"),
            horizon=getattr(config, "horizon", model.horizon),
            suffix=suffix,
        )

        result_dir = Path(__file__).resolve().parents[1] / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        learning_tbl = result_dir / (
            f"mdc_learning_constraint_violations_"
            f"{getattr(config, 'method', 'cls')}{getattr(config, 'penalty', 'na')}_"
            f"T{getattr(config, 'horizon', model.horizon)}{suffix}.csv"
        )
        _save_violation_count_table(constraint_labels, policy_viol_count, learning_tbl)
        print(f"[Multi-DC] Saved learning violation table: {learning_tbl}")

    if len(policy_schedule_rows) > 0:
        suffix = "-g" if bool(getattr(config, "penalty_growth", False)) else ""
        result_dir = Path(__file__).resolve().parents[1] / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        schedule_tbl = result_dir / (
            f"mdc_policy_schedule_"
            f"{getattr(config, 'method', 'cls')}{getattr(config, 'penalty', 'na')}_"
            f"T{getattr(config, 'horizon', model.horizon)}{suffix}.csv"
        )
        pd.DataFrame(policy_schedule_rows).to_csv(schedule_tbl, index=False)
        print(f"[Multi-DC] Saved policy decision-variable schedule: {schedule_tbl}")

    return df

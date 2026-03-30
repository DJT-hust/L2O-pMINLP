"""
Neuromancer-style penalty loss for Microgrid Scheduling (MIQP).

Flattened decision vector x (length 9T) excludes p_grid and soc:
- p_grid is computed from buy/sell
- soc is derived from charge/discharge dynamics via _reconstruct_soc()

x = [p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis]

All tensors are expected to be batched: shape [B, T] or [B, 9T].
"""

from __future__ import annotations

import torch
from torch import nn


class penaltyLoss(nn.Module):
    def __init__(
        self,
        input_keys,
        horizon: int,
        penalty_weight: float = 50.0,
        # limits / coefficients (should match math_solver defaults unless you override)
        p_gen_max: float = 2.0,
        p_ch_max: float = 1.0,
        p_dis_max: float = 1.0,
        soc_min: float = 0.1,
        soc_max: float = 4.0,
        eta_ch: float = 0.95,
        eta_dis: float = 0.95,
        gen_quad: float = 0.06,
        gen_lin: float = 0.1,
        gen_on_cost: float = 0.2,
        load_shed_penalty: float = 10.0,
        output_key: str = "loss",
        eq_weight: float = 1.0,
        obj_weight: float = 1.0,
        viol_weight: float = 1.0,
        viol_threshold: float = 0.0,
        distill_weight: float = 0.0,
        teacher_key: str = "x_teacher",
        teacher_mask_key: str = "teacher_mask",
    ):
        super().__init__()
        assert len(input_keys) == 6, "Expected input_keys = ['load','pv','price_buy','price_sell','soc0','x_key']"

        self.horizon = int(horizon)
        self.output_key = output_key
        self.penalty_weight = float(penalty_weight)
        self.eq_weight = float(eq_weight)
        self.obj_weight = float(obj_weight)
        self.viol_weight = float(viol_weight)
        self.viol_threshold = float(viol_threshold)
        self.distill_weight = float(distill_weight)
        self.teacher_key = teacher_key
        self.teacher_mask_key = teacher_mask_key

        self.load_key, self.pv_key, self.price_buy_key, self.price_sell_key, self.soc0_key, self.x_key = input_keys

        # params
        self.p_gen_max = float(p_gen_max)
        self.p_ch_max = float(p_ch_max)
        self.p_dis_max = float(p_dis_max)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.eta_ch = float(eta_ch)
        self.eta_dis = float(eta_dis)

        # costs
        self.gen_quad = float(gen_quad)
        self.gen_lin = float(gen_lin)
        self.gen_on_cost = float(gen_on_cost)
        self.load_shed_penalty = float(load_shed_penalty)

    def _unpack_x(self, x: torch.Tensor):
        """
        Unpack flattened decision vector x (shape [B, 9T]) into individual variables.
        
        x = [p_grid_buy(0:T), p_grid_sell(T:2T), p_gen(2T:3T), p_ch(3T:4T), 
             p_dis(4T:5T), s_load(5T:6T), u_gen(6T:7T), u_ch(7T:8T), u_dis(8T:9T)]
        
        SOC is NOT included in x; it is derived from dynamics.
        p_grid is computed as p_grid_buy - p_grid_sell.
        """
        B, n = x.shape
        T = self.horizon
        assert n == 9 * T, f"Expected x dim {9*T}, got {n}"

        def sl(a, b):
            return x[:, a * T : b * T]

        p_grid_buy = sl(0, 1)
        p_grid_sell = sl(1, 2)
        p_gen = sl(2, 3)
        p_ch = sl(3, 4)
        p_dis = sl(4, 5)
        s_load = sl(5, 6)
        u_gen = sl(6, 7)
        u_ch = sl(7, 8)
        u_dis = sl(8, 9)
        return p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis

    def _reconstruct_soc(self, p_ch: torch.Tensor, p_dis: torch.Tensor, soc0: torch.Tensor):
        """
        Build SOC deterministically from dynamics, treating SOC as an implicit variable.
        """
        T = self.horizon
        soc = []
        soc_prev = soc0 + self.eta_ch * p_ch[:, 0] - (1.0 / self.eta_dis) * p_dis[:, 0]
        soc.append(soc_prev)
        for t in range(1, T):
            soc_prev = soc_prev + self.eta_ch * p_ch[:, t] - (1.0 / self.eta_dis) * p_dis[:, t]
            soc.append(soc_prev)
        return torch.stack(soc, dim=1)

    def cal_obj(self, input_dict):
        x = input_dict[self.x_key]
        price_buy = input_dict[self.price_buy_key]
        price_sell = input_dict[self.price_sell_key]
        p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis = self._unpack_x(x)

        elec_cost = (price_buy * p_grid_buy - price_sell * p_grid_sell).sum(dim=1)
        gen_cost = (self.gen_quad * (p_gen ** 2) + self.gen_lin * p_gen + self.gen_on_cost * u_gen).sum(dim=1)
        shed_cost = (self.load_shed_penalty * s_load).sum(dim=1)
        return elec_cost + gen_cost + shed_cost

    def cal_constr_viol(self, input_dict):
        """
        Calculate constraint violations including power_balance penalty.

        NOTE on power_balance strategy:
        - We keep a mild penalty on power_balance (weight=0.1) during training.
        - This prevents the network from producing arbitrarily infeasible solutions.
        - Stronger enforcement is done in the projection step.

        SOC is not learned; it is derived from charge/discharge dynamics.
        ch_dis_mutex gets double penalty to strengthen enforcement.
        """
        x = input_dict[self.x_key]
        load = input_dict[self.load_key]
        pv = input_dict[self.pv_key]
        soc0 = input_dict[self.soc0_key]

        p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis = self._unpack_x(x)
        p_grid = p_grid_buy - p_grid_sell

        # soc0 is recommended shape [B,1]
        if soc0.ndim == 2 and soc0.shape[1] == 1:
            soc0v = soc0[:, 0]
        else:
            soc0v = soc0

        # Reconstruct SOC directly from dynamics
        soc = self._reconstruct_soc(p_ch, p_dis, soc0v)

        relu = torch.relu
        
        # POWER BALANCE penalty
        res_balance = p_gen + p_dis - p_ch + p_grid + pv + s_load - load
        viol_eq = (res_balance ** 2).sum(dim=1) * self.eq_weight

        viol_ineq = torch.zeros_like(viol_eq)

        # nonnegativity (important for network outputs)
        viol_ineq = viol_ineq + relu(-p_grid_buy).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_grid_sell).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_gen).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_ch).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_dis).sum(dim=1)
        viol_ineq = viol_ineq + relu(-s_load).sum(dim=1)

        # generator bound linked with commitment
        viol_ineq = viol_ineq + relu(p_gen - self.p_gen_max * u_gen).sum(dim=1)

        # battery bounds - CRITICAL: apply 20x weight for ch/dis capacity constraints
        viol_ineq = viol_ineq + 20.0 * relu(p_ch - self.p_ch_max * u_ch).sum(dim=1)
        viol_ineq = viol_ineq + 20.0 * relu(p_dis - self.p_dis_max * u_dis).sum(dim=1)
        
        # CHARGE-DISCHARGE MUTUAL EXCLUSION: 3x PENALTY for stronger enforcement
        ch_dis_violation = relu(u_ch + u_dis - 1.0).sum(dim=1)
        viol_ineq = viol_ineq + 3.0 * ch_dis_violation  # increased from 2x
        
        # SOC bounds - 2x weight for better state management
        viol_ineq = viol_ineq + 2.0 * relu(self.soc_min - soc).sum(dim=1)
        viol_ineq = viol_ineq + 2.0 * relu(soc - self.soc_max).sum(dim=1)

        return viol_eq + viol_ineq

    def forward(self, input_dict):
        obj = self.cal_obj(input_dict)
        viol = self.cal_constr_viol(input_dict)
        # Hinge on violation keeps training focused on objective once near-feasible.
        viol_term = torch.relu(viol - self.viol_threshold)
        loss = self.obj_weight * obj + self.penalty_weight * self.viol_weight * viol_term

        # Optional solver-guided distillation on labeled subset.
        if self.distill_weight > 0.0 and self.teacher_key in input_dict:
            x_pred = input_dict[self.x_key]
            x_teacher = input_dict[self.teacher_key]
            mse_per_sample = ((x_pred - x_teacher) ** 2).mean(dim=1)
            if self.teacher_mask_key in input_dict:
                mask = input_dict[self.teacher_mask_key].reshape(-1).float()
                denom = torch.clamp(mask.sum(), min=1.0)
                distill_term = (mse_per_sample * mask).sum() / denom
            else:
                distill_term = mse_per_sample.mean()
            loss = loss + self.distill_weight * distill_term

        input_dict[self.output_key] = torch.mean(loss)
        return input_dict
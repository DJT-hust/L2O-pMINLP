"""
Neuromancer-style penalty loss for Microgrid Scheduling (MIQP).

Flattened decision vector x (length 10T) excludes p_grid:
x = [p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, soc, s_load, u_gen, u_ch, u_dis]

All tensors are expected to be batched: shape [B, T] or [B, 10T].
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
    ):
        """
        input_keys: list/tuple of keys in input_dict, e.g.
            ["load", "pv", "price_buy", "price_sell", "soc0", "x_rnd"]
        horizon: T
        penalty_weight: rho for (total) constraint violation
        eq_weight: scaling for equality residual terms (squared)
        """
        super().__init__()
        assert len(input_keys) >= 2, "input_keys must include parameters and a decision key"

        self.horizon = int(horizon)
        self.output_key = output_key
        self.penalty_weight = float(penalty_weight)
        self.eq_weight = float(eq_weight)

        # keys
        # expected: load, pv, price_buy, price_sell, soc0, x_key
        self.load_key = input_keys[0]
        self.pv_key = input_keys[1]
        self.price_buy_key = input_keys[2]
        self.price_sell_key = input_keys[3]
        self.soc0_key = input_keys[4]
        self.x_key = input_keys[5]

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

    # --------------------------
    # helpers: unpack x
    # --------------------------
    def _unpack_x(self, x: torch.Tensor):
        """
        x: [B, 10T]
        returns each block: [B, T]
        """
        B, n = x.shape
        T = self.horizon
        assert n == 10 * T, f"Expected x dim {10*T}, got {n}"

        def sl(a, b):
            return x[:, a * T : b * T]

        p_grid_buy = sl(0, 1)
        p_grid_sell = sl(1, 2)
        p_gen = sl(2, 3)
        p_ch = sl(3, 4)
        p_dis = sl(4, 5)
        soc = sl(5, 6)
        s_load = sl(6, 7)
        u_gen = sl(7, 8)
        u_ch = sl(8, 9)
        u_dis = sl(9, 10)

        return p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, soc, s_load, u_gen, u_ch, u_dis

    # --------------------------
    # objective
    # --------------------------
    def cal_obj(self, input_dict):
        x = input_dict[self.x_key]
        load = input_dict[self.load_key]
        pv = input_dict[self.pv_key]
        price_buy = input_dict[self.price_buy_key]
        price_sell = input_dict[self.price_sell_key]
        soc0 = input_dict[self.soc0_key]

        # shape checks (loose)
        T = self.horizon
        assert load.shape[-1] == T and pv.shape[-1] == T and price_buy.shape[-1] == T and price_sell.shape[-1] == T

        p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, soc, s_load, u_gen, u_ch, u_dis = self._unpack_x(x)

        elec_cost = (price_buy * p_grid_buy - price_sell * p_grid_sell).sum(dim=1)
        gen_cost = (self.gen_quad * (p_gen ** 2) + self.gen_lin * p_gen + self.gen_on_cost * u_gen).sum(dim=1)
        shed_cost = (self.load_shed_penalty * s_load).sum(dim=1)

        return elec_cost + gen_cost + shed_cost

    # --------------------------
    # constraint violation (differentiable)
    # --------------------------
    def cal_constr_viol(self, input_dict):
        x = input_dict[self.x_key]
        load = input_dict[self.load_key]
        pv = input_dict[self.pv_key]
        soc0 = input_dict[self.soc0_key]

        p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, soc, s_load, u_gen, u_ch, u_dis = self._unpack_x(x)

        # derived net exchange
        p_grid = p_grid_buy - p_grid_sell

        # equality residuals (use squared residuals)
        # power balance: p_gen + p_dis - p_ch + p_grid + pv + s_load == load
        res_balance = p_gen + p_dis - p_ch + p_grid + pv + s_load - load  # [B,T]
        viol_eq = (res_balance ** 2).sum(dim=1) * self.eq_weight

        # SOC dynamics
        # soc[0] == soc0 + eta_ch*p_ch[0] - (1/eta_dis)*p_dis[0]
        res_soc0 = soc[:, 0] - (soc0.squeeze(-1) if soc0.ndim == 2 and soc0.shape[1] == 1 else soc0) \
                   - self.eta_ch * p_ch[:, 0] + (1.0 / self.eta_dis) * p_dis[:, 0]
        viol_eq = viol_eq + (res_soc0 ** 2) * self.eq_weight

        # soc[t] == soc[t-1] + eta_ch*p_ch[t] - (1/eta_dis)*p_dis[t], t>=1
        if self.horizon > 1:
            res_soc = soc[:, 1:] - soc[:, :-1] - self.eta_ch * p_ch[:, 1:] + (1.0 / self.eta_dis) * p_dis[:, 1:]
            viol_eq = viol_eq + (res_soc ** 2).sum(dim=1) * self.eq_weight

        # inequality violations (use ReLU)
        relu = torch.relu
        viol_ineq = torch.zeros_like(viol_eq)

        # nonnegativity (helpful because network output can be negative)
        viol_ineq = viol_ineq + relu(-p_grid_buy).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_grid_sell).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_gen).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_ch).sum(dim=1)
        viol_ineq = viol_ineq + relu(-p_dis).sum(dim=1)
        viol_ineq = viol_ineq + relu(-s_load).sum(dim=1)

        # generator bound linked with commitment: p_gen <= p_gen_max*u_gen
        viol_ineq = viol_ineq + relu(p_gen - self.p_gen_max * u_gen).sum(dim=1)

        # battery bounds: p_ch <= p_ch_max*u_ch; p_dis <= p_dis_max*u_dis
        viol_ineq = viol_ineq + relu(p_ch - self.p_ch_max * u_ch).sum(dim=1)
        viol_ineq = viol_ineq + relu(p_dis - self.p_dis_max * u_dis).sum(dim=1)

        # exclusivity: u_ch + u_dis <= 1
        viol_ineq = viol_ineq + relu(u_ch + u_dis - 1.0).sum(dim=1)

        # SOC bounds: soc_min <= soc <= soc_max
        viol_ineq = viol_ineq + relu(self.soc_min - soc).sum(dim=1)
        viol_ineq = viol_ineq + relu(soc - self.soc_max).sum(dim=1)

        # total violation per sample
        return viol_eq + viol_ineq

    def forward(self, input_dict):
        obj = self.cal_obj(input_dict)
        viol = self.cal_constr_viol(input_dict)
        loss = obj + self.penalty_weight * viol
        input_dict[self.output_key] = torch.mean(loss)
        return input_dict
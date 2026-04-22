"""
Pre-defined neural network layers
"""
import torch
from torch import nn


def _multidc_split_xi(xi, horizon, num_dc, num_regions, num_jobs):
    t = int(horizon)
    d = int(num_dc)
    r = int(num_regions)
    j = int(num_jobs)

    idx = 0
    base_load = xi[:, idx : idx + t]
    idx += t
    price = xi[:, idx : idx + t]
    idx += t

    renew = xi[:, idx : idx + d * t].reshape(-1, d, t)
    idx += d * t
    interactive = xi[:, idx : idx + r * t].reshape(-1, r, t)
    idx += r * t
    batch_work = xi[:, idx : idx + j]
    return base_load, price, renew, interactive, batch_work


def _multidc_constraint_features(base_load, price, renew, interactive, batch_work):
    """Constraint-aware per-timestep engineered features for Multi-DC policies."""
    interactive_total = interactive.sum(dim=1)
    renew_total = renew.sum(dim=1)
    net_power_pressure = base_load + interactive_total - renew_total
    price_weighted_pressure = price * net_power_pressure
    batch_total_rep = batch_work.sum(dim=1, keepdim=True).repeat(1, base_load.shape[1])

    return (
        interactive_total,
        renew_total,
        net_power_pressure,
        price_weighted_pressure,
        batch_total_rep,
    )


class MultiDCMLPPolicy(nn.Module):
    """MLP policy with optional constraint-aware feature injection."""

    def __init__(
        self,
        horizon,
        num_dc,
        num_regions,
        num_jobs,
        input_dim,
        hidden_dim,
        depth,
        out_dim,
        dropout=0.2,
        constraint_feat_inject=False,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)
        self.constraint_feat_inject = bool(constraint_feat_inject)

        extra_dim = 5 * self.horizon if self.constraint_feat_inject else 0
        d = int(input_dim) + extra_dim

        layers = []
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            d = int(hidden_dim)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def _augment(self, xi):
        if not self.constraint_feat_inject:
            return xi
        base_load, price, renew, interactive, batch_work = _multidc_split_xi(
            xi, self.horizon, self.num_dc, self.num_regions, self.num_jobs
        )
        feats = _multidc_constraint_features(base_load, price, renew, interactive, batch_work)
        feat_flat = torch.cat(list(feats), dim=1)
        return torch.cat([xi, feat_flat], dim=1)

    def forward(self, xi):
        return self.net(self._augment(xi))


class TemporalResidualBlock(nn.Module):
    """Dilated Conv1d residual block for sequence modeling."""

    def __init__(self, hidden_dim, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return self.act(x + res)


class DualHeadTemporalResidualPolicy(nn.Module):
    """
    Microgrid policy with:
    - temporal backbone (TCN-style Conv1d residual blocks),
    - dual heads (feasibility/objective),
    - residual strategy on top of a simple heuristic baseline.
    """

    def __init__(
        self,
        horizon,
        out_dim,
        hidden_dim=128,
        num_blocks=4,
        dropout=0.1,
        residual_scale=0.6,
        refine_scale=0.15,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = max(1, int(num_blocks))
        self.residual_scale = float(residual_scale)
        self.refine_scale = float(refine_scale)

        # channels: [load, pv, price_buy, price_sell, soc0_broadcast]
        self.input_proj = nn.Conv1d(5, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    self.hidden_dim,
                    dilation=2 ** (i % 4),
                    dropout=dropout,
                )
                for i in range(self.num_blocks)
            ]
        )

        flat_dim = self.hidden_dim * self.horizon
        self.feas_head = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.cross_refiner = nn.Sequential(
            nn.Linear(self.out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _split_xi(self, xi):
        t = self.horizon
        load = xi[:, 0:t]
        pv = xi[:, t:2 * t]
        price_buy = xi[:, 2 * t:3 * t]
        price_sell = xi[:, 3 * t:4 * t]
        soc0 = xi[:, 4 * t:4 * t + 1]
        return load, pv, price_buy, price_sell, soc0

    def _heuristic_baseline(self, xi):
        """
        Build a conservative baseline dispatch in the flattened x layout:
        [p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis].
        """
        load, pv, price_buy, _, _ = self._split_xi(xi)
        t = self.horizon
        bsz = xi.shape[0]

        p_grid_buy = torch.relu(load - pv)
        p_grid_sell = torch.relu(pv - load)

        # low-price charging / high-price discharging prior.
        buy_mean = price_buy.mean(dim=1, keepdim=True)
        low_price = (price_buy < buy_mean).float()
        high_price = (price_buy > buy_mean).float()

        p_gen = torch.zeros(bsz, t, device=xi.device, dtype=xi.dtype)
        p_ch = 0.15 * low_price
        p_dis = 0.15 * high_price
        s_load = torch.zeros(bsz, t, device=xi.device, dtype=xi.dtype)
        u_gen = torch.zeros(bsz, t, device=xi.device, dtype=xi.dtype)
        u_ch = low_price
        u_dis = high_price

        return torch.cat([p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis], dim=1)

    def forward(self, xi):
        load, pv, price_buy, price_sell, soc0 = self._split_xi(xi)
        soc_seq = soc0.repeat(1, self.horizon)

        seq = torch.stack([load, pv, price_buy, price_sell, soc_seq], dim=1)  # [B,5,T]
        h = self.input_proj(seq)
        for block in self.blocks:
            h = block(h)

        h_flat = h.reshape(h.shape[0], -1)
        h_global = h.mean(dim=-1)

        feas_delta = self.feas_head(h_flat)
        obj_delta = self.obj_head(h_flat)
        gate = torch.sigmoid(self.gate_head(h_global))

        # Gate interpolates between feasibility and objective experts.
        delta = (1.0 - gate) * feas_delta + gate * obj_delta
        delta = self.residual_scale * torch.tanh(delta)

        x_base = self._heuristic_baseline(xi)
        if x_base.shape[1] != self.out_dim:
            x_base = torch.zeros(xi.shape[0], self.out_dim, device=xi.device, dtype=xi.dtype)

        x_pre = x_base + delta
        if self.refine_scale != 0.0:
            refine = self.refine_scale * torch.tanh(self.cross_refiner(x_pre))
            return x_pre + refine
        return x_pre


class DualHeadHybridTemporalPolicy(nn.Module):
    """
    Hybrid temporal policy with local TCN branch + global Transformer branch.
    It keeps the dual-head residual decoding used by the original temporal policy.
    """

    def __init__(
        self,
        horizon,
        out_dim,
        hidden_dim=128,
        num_blocks=4,
        dropout=0.1,
        residual_scale=0.6,
        refine_scale=0.15,
        tf_layers=2,
        tf_heads=4,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = max(1, int(num_blocks))
        self.residual_scale = float(residual_scale)
        self.refine_scale = float(refine_scale)

        # Local branch (TCN style)
        self.local_input_proj = nn.Conv1d(5, self.hidden_dim, kernel_size=1)
        self.local_blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    self.hidden_dim,
                    dilation=2 ** (i % 4),
                    dropout=dropout,
                )
                for i in range(self.num_blocks)
            ]
        )

        # Global branch (Transformer over time tokens)
        self.global_input_proj = nn.Linear(5, self.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.horizon, self.hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=max(1, int(tf_heads)),
            dim_feedforward=4 * self.hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(tf_layers)))

        # Fuse local/global temporal features.
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        flat_dim = self.hidden_dim * self.horizon
        self.feas_head = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.cross_refiner = nn.Sequential(
            nn.Linear(self.out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _split_xi(self, xi):
        t = self.horizon
        load = xi[:, 0:t]
        pv = xi[:, t:2 * t]
        price_buy = xi[:, 2 * t:3 * t]
        price_sell = xi[:, 3 * t:4 * t]
        soc0 = xi[:, 4 * t:4 * t + 1]
        return load, pv, price_buy, price_sell, soc0

    def _heuristic_baseline(self, xi):
        load, pv, price_buy, _, _ = self._split_xi(xi)
        t = self.horizon
        bsz = xi.shape[0]

        p_grid_buy = torch.relu(load - pv)
        p_grid_sell = torch.relu(pv - load)

        buy_mean = price_buy.mean(dim=1, keepdim=True)
        low_price = (price_buy < buy_mean).float()
        high_price = (price_buy > buy_mean).float()

        p_gen = torch.zeros(bsz, t, device=xi.device, dtype=xi.dtype)
        p_ch = 0.15 * low_price
        p_dis = 0.15 * high_price
        s_load = torch.zeros(bsz, t, device=xi.device, dtype=xi.dtype)
        u_gen = torch.zeros(bsz, t, device=xi.device, dtype=xi.dtype)
        u_ch = low_price
        u_dis = high_price

        return torch.cat([p_grid_buy, p_grid_sell, p_gen, p_ch, p_dis, s_load, u_gen, u_ch, u_dis], dim=1)

    def forward(self, xi):
        load, pv, price_buy, price_sell, soc0 = self._split_xi(xi)
        soc_seq = soc0.repeat(1, self.horizon)
        seq = torch.stack([load, pv, price_buy, price_sell, soc_seq], dim=1)  # [B,5,T]

        # Local features
        h_local = self.local_input_proj(seq)
        for block in self.local_blocks:
            h_local = block(h_local)

        # Global features
        tokens = seq.transpose(1, 2)  # [B,T,5]
        h_global = self.global_input_proj(tokens)
        h_global = self.transformer(h_global + self.pos_emb)
        h_global = h_global.transpose(1, 2)  # [B,H,T]

        # Fuse and decode
        h = self.fuse(torch.cat([h_local, h_global], dim=1))
        h_flat = h.reshape(h.shape[0], -1)
        h_pool = h.mean(dim=-1)

        feas_delta = self.feas_head(h_flat)
        obj_delta = self.obj_head(h_flat)
        gate = torch.sigmoid(self.gate_head(h_pool))

        delta = (1.0 - gate) * feas_delta + gate * obj_delta
        delta = self.residual_scale * torch.tanh(delta)

        x_base = self._heuristic_baseline(xi)
        if x_base.shape[1] != self.out_dim:
            x_base = torch.zeros(xi.shape[0], self.out_dim, device=xi.device, dtype=xi.dtype)

        x_pre = x_base + delta
        if self.refine_scale != 0.0:
            refine = self.refine_scale * torch.tanh(self.cross_refiner(x_pre))
            return x_pre + refine
        return x_pre


class MultiDCLSTMPolicy(nn.Module):
    """LSTM policy for flattened Multi-DC inputs."""

    def __init__(
        self,
        horizon,
        num_dc,
        num_regions,
        num_jobs,
        out_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        constraint_feat_inject=False,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.constraint_feat_inject = bool(constraint_feat_inject)

        self.seq_in_dim = 2 + self.num_dc + self.num_regions + self.num_jobs
        if self.constraint_feat_inject:
            self.seq_in_dim += 5
        self.in_proj = nn.Linear(self.seq_in_dim, self.hidden_dim)

        layers = max(1, int(num_layers))
        lstm_dropout = float(dropout) if layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * self.horizon, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _split_xi(self, xi):
        return _multidc_split_xi(xi, self.horizon, self.num_dc, self.num_regions, self.num_jobs)

    def forward(self, xi):
        base_load, price, renew, interactive, batch_work = self._split_xi(xi)

        # Build per-timestep features: [base_load, price, renew(D), interactive(R), batch_work(J)].
        renew_t = renew.transpose(1, 2)
        interactive_t = interactive.transpose(1, 2)
        batch_t = batch_work.unsqueeze(1).repeat(1, self.horizon, 1)
        seq = torch.cat(
            [
                base_load.unsqueeze(-1),
                price.unsqueeze(-1),
                renew_t,
                interactive_t,
                batch_t,
            ],
            dim=-1,
        )

        if self.constraint_feat_inject:
            feats = _multidc_constraint_features(base_load, price, renew, interactive, batch_work)
            feat_seq = torch.stack(feats, dim=-1)
            seq = torch.cat([seq, feat_seq], dim=-1)

        h = torch.relu(self.in_proj(seq))
        h, _ = self.lstm(h)
        h = h.reshape(h.shape[0], -1)
        return self.head(h)


class MultiDCRNNPolicy(nn.Module):
    """Vanilla RNN policy for flattened Multi-DC inputs."""

    def __init__(
        self,
        horizon,
        num_dc,
        num_regions,
        num_jobs,
        out_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        constraint_feat_inject=False,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.constraint_feat_inject = bool(constraint_feat_inject)

        self.seq_in_dim = 2 + self.num_dc + self.num_regions + self.num_jobs
        if self.constraint_feat_inject:
            self.seq_in_dim += 5
        self.in_proj = nn.Linear(self.seq_in_dim, self.hidden_dim)

        layers = max(1, int(num_layers))
        rnn_dropout = float(dropout) if layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=rnn_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * self.horizon, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _split_xi(self, xi):
        return _multidc_split_xi(xi, self.horizon, self.num_dc, self.num_regions, self.num_jobs)

    def forward(self, xi):
        base_load, price, renew, interactive, batch_work = self._split_xi(xi)

        renew_t = renew.transpose(1, 2)
        interactive_t = interactive.transpose(1, 2)
        batch_t = batch_work.unsqueeze(1).repeat(1, self.horizon, 1)
        seq = torch.cat(
            [
                base_load.unsqueeze(-1),
                price.unsqueeze(-1),
                renew_t,
                interactive_t,
                batch_t,
            ],
            dim=-1,
        )

        if self.constraint_feat_inject:
            feats = _multidc_constraint_features(base_load, price, renew, interactive, batch_work)
            feat_seq = torch.stack(feats, dim=-1)
            seq = torch.cat([seq, feat_seq], dim=-1)

        h = torch.relu(self.in_proj(seq))
        h, _ = self.rnn(h)
        h = h.reshape(h.shape[0], -1)
        return self.head(h)


class MultiDCTCNPolicy(nn.Module):
    """TCN-style policy for flattened Multi-DC inputs."""

    def __init__(
        self,
        horizon,
        num_dc,
        num_regions,
        num_jobs,
        out_dim,
        hidden_dim=128,
        num_blocks=4,
        dropout=0.1,
        constraint_feat_inject=False,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.constraint_feat_inject = bool(constraint_feat_inject)

        in_ch = 2 + self.num_dc + self.num_regions + self.num_jobs
        if self.constraint_feat_inject:
            in_ch += 5
        self.input_proj = nn.Conv1d(in_ch, self.hidden_dim, kernel_size=1)
        blocks = max(1, int(num_blocks))
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    self.hidden_dim,
                    dilation=2 ** (i % 4),
                    dropout=float(dropout),
                )
                for i in range(blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * self.horizon, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def _split_xi(self, xi):
        return _multidc_split_xi(xi, self.horizon, self.num_dc, self.num_regions, self.num_jobs)

    def forward(self, xi):
        base_load, price, renew, interactive, batch_work = self._split_xi(xi)

        renew_t = renew.transpose(1, 2)
        interactive_t = interactive.transpose(1, 2)
        batch_t = batch_work.unsqueeze(1).repeat(1, self.horizon, 1)
        seq = torch.cat(
            [
                base_load.unsqueeze(-1),
                price.unsqueeze(-1),
                renew_t,
                interactive_t,
                batch_t,
            ],
            dim=-1,
        )

        if self.constraint_feat_inject:
            feats = _multidc_constraint_features(base_load, price, renew, interactive, batch_work)
            feat_seq = torch.stack(feats, dim=-1)
            seq = torch.cat([seq, feat_seq], dim=-1)
        # [B,T,C] -> [B,C,T]
        h = seq.transpose(1, 2)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = h.reshape(h.shape[0], -1)
        return self.head(h)


class MultiDCDualHeadMLPPolicy(nn.Module):
    """Dual-head gated MLP policy with optional residual baseline."""

    def __init__(
        self,
        input_dim,
        out_dim,
        hidden_dim=128,
        depth=6,
        dropout=0.2,
        residual=False,
        residual_scale=0.3,
        horizon=288,
        num_dc=3,
        num_regions=3,
        num_jobs=4,
        constraint_feat_inject=False,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)
        self.constraint_feat_inject = bool(constraint_feat_inject)

        extra_dim = 5 * self.horizon if self.constraint_feat_inject else 0
        layers = []
        d = int(input_dim) + extra_dim
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            d = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)

        self.feas_head = nn.Linear(d, self.out_dim)
        self.obj_head = nn.Linear(d, self.out_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(d, max(8, d // 4)),
            nn.ReLU(),
            nn.Linear(max(8, d // 4), 1),
        )

        if self.residual:
            self.base_head = nn.Sequential(
                nn.Linear(int(input_dim) + extra_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), self.out_dim),
            )
        else:
            self.base_head = None

    def _augment(self, xi):
        if not self.constraint_feat_inject:
            return xi
        base_load, price, renew, interactive, batch_work = _multidc_split_xi(
            xi, self.horizon, self.num_dc, self.num_regions, self.num_jobs
        )
        feats = _multidc_constraint_features(base_load, price, renew, interactive, batch_work)
        feat_flat = torch.cat(list(feats), dim=1)
        return torch.cat([xi, feat_flat], dim=1)

    def forward(self, xi):
        x_aug = self._augment(xi)
        h = self.backbone(x_aug)
        feas = self.feas_head(h)
        obj = self.obj_head(h)
        gate = torch.sigmoid(self.gate_head(h))
        delta = (1.0 - gate) * feas + gate * obj

        if self.base_head is None:
            return delta
        base = self.base_head(x_aug)
        return base + self.residual_scale * torch.tanh(delta)


class MultiDCDualHeadTCNPolicy(nn.Module):
    """Dual-head gated TCN policy with optional residual baseline."""

    def __init__(
        self,
        horizon,
        num_dc,
        num_regions,
        num_jobs,
        out_dim,
        hidden_dim=128,
        num_blocks=4,
        dropout=0.1,
        residual=False,
        residual_scale=0.3,
        constraint_feat_inject=False,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_dc = int(num_dc)
        self.num_regions = int(num_regions)
        self.num_jobs = int(num_jobs)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.constraint_feat_inject = bool(constraint_feat_inject)

        in_ch = 2 + self.num_dc + self.num_regions + self.num_jobs
        if self.constraint_feat_inject:
            in_ch += 5
        self.input_proj = nn.Conv1d(in_ch, self.hidden_dim, kernel_size=1)
        blocks = max(1, int(num_blocks))
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    self.hidden_dim,
                    dilation=2 ** (i % 4),
                    dropout=float(dropout),
                )
                for i in range(blocks)
            ]
        )

        flat_dim = self.hidden_dim * self.horizon
        self.feas_head = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, max(8, self.hidden_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(8, self.hidden_dim // 4), 1),
        )

        if self.residual:
            in_dim = 2 * self.horizon + self.num_dc * self.horizon + self.num_regions * self.horizon + self.num_jobs
            self.base_head = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.base_head = None

    def _split_xi(self, xi):
        return _multidc_split_xi(xi, self.horizon, self.num_dc, self.num_regions, self.num_jobs)

    def forward(self, xi):
        base_load, price, renew, interactive, batch_work = self._split_xi(xi)

        renew_t = renew.transpose(1, 2)
        interactive_t = interactive.transpose(1, 2)
        batch_t = batch_work.unsqueeze(1).repeat(1, self.horizon, 1)
        seq = torch.cat(
            [
                base_load.unsqueeze(-1),
                price.unsqueeze(-1),
                renew_t,
                interactive_t,
                batch_t,
            ],
            dim=-1,
        )

        if self.constraint_feat_inject:
            feats = _multidc_constraint_features(base_load, price, renew, interactive, batch_work)
            feat_seq = torch.stack(feats, dim=-1)
            seq = torch.cat([seq, feat_seq], dim=-1)

        h = seq.transpose(1, 2)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)

        h_flat = h.reshape(h.shape[0], -1)
        h_pool = h.mean(dim=-1)
        feas = self.feas_head(h_flat)
        obj = self.obj_head(h_flat)
        gate = torch.sigmoid(self.gate_head(h_pool))
        delta = (1.0 - gate) * feas + gate * obj

        if self.base_head is None:
            return delta
        base = self.base_head(xi)
        return base + self.residual_scale * torch.tanh(delta)

class netFC(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Fully connected neural network with configurable dimensions.
        """
        super(netFC, self).__init__()
        # build network layer structure
        self.layers = nn.ModuleList()
        sizes = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(sizes) - 2):
            self.layers.append(layerFC(sizes[i], sizes[i + 1]))
        # last layer without ReLU, BatchNorm, and Dropout
        self.layers.append(nn.Linear(sizes[-2], sizes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class layerFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(layerFC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = self.fc(x)
        h = self.relu(h)
        h = self.bn(h)
        h = self.dropout(h)
        return h


if __name__ == "__main__":
    import torch

    # random seed
    torch.manual_seed(42)

    # initialize the model
    layer = netFC(input_dim=10, hidden_dims=[20,20], output_dim=10)
    print(layer)

    # generate random input data: batch size = 32
    input_data = torch.randn(32, 10)

    # test the forward pass
    output_data = layer(input_data)
    print(output_data[0])

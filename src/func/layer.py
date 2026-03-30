"""
Pre-defined neural network layers
"""
import torch
from torch import nn


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

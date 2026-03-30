"""
Projection Gradient
"""

import numpy as np
import torch
from torch import nn
import neuromancer as nm

class gradientProjection(nn.Module):
    def __init__(
        self,
        pre_components,
        post_components,
        loss_fn,
        target_key,
        max_iters=1000,
        step_size=0.01,
        decay=1.0,
        obj_guard_weight=0.0,
        viol_tol=1e-6,
        early_stop_patience=20,
        min_viol_improve=1e-8,
        restarts=1,
        restart_noise_std=0.0,
        tail_steps=0,
        tail_weight=1.0,
        end_discharge_steps=0,
        end_discharge_scale=1.0,
    ):
        super().__init__()
        self.pre_components = pre_components
        self.post_components = post_components
        self.loss_fn = loss_fn
        self.target_key = target_key
        self.max_iters = max_iters
        self.step_size = step_size
        self.decay = decay
        self.obj_guard_weight = float(obj_guard_weight)
        self.viol_tol = float(viol_tol)
        self.early_stop_patience = int(early_stop_patience)
        self.min_viol_improve = float(min_viol_improve)
        self.restarts = max(1, int(restarts))
        self.restart_noise_std = float(restart_noise_std)
        self.tail_steps = max(0, int(tail_steps))
        self.tail_weight = float(tail_weight)
        self.end_discharge_steps = max(0, int(end_discharge_steps))
        self.end_discharge_scale = float(end_discharge_scale)

    def _tail_weighted_obj(self, input_dict):
        """
        Objective guard with optional extra weight on last K timesteps.
        Falls back to standard objective when tail weighting is disabled.
        """
        obj = self.loss_fn.cal_obj(input_dict)
        if self.tail_steps <= 0 or self.tail_weight <= 1.0:
            return obj

        if not hasattr(self.loss_fn, "_unpack_x"):
            return obj

        x = input_dict[self.loss_fn.x_key]
        price_buy = input_dict[self.loss_fn.price_buy_key]
        price_sell = input_dict[self.loss_fn.price_sell_key]
        p_grid_buy, p_grid_sell, p_gen, _, _, s_load, u_gen, _, _ = self.loss_fn._unpack_x(x)

        t = self.loss_fn.horizon
        k = min(self.tail_steps, t)
        tail = slice(t - k, t)

        elec_tail = (price_buy[:, tail] * p_grid_buy[:, tail] - price_sell[:, tail] * p_grid_sell[:, tail]).sum(dim=1)
        gen_tail = (
            self.loss_fn.gen_quad * (p_gen[:, tail] ** 2)
            + self.loss_fn.gen_lin * p_gen[:, tail]
            + self.loss_fn.gen_on_cost * u_gen[:, tail]
        ).sum(dim=1)
        shed_tail = (self.loss_fn.load_shed_penalty * s_load[:, tail]).sum(dim=1)
        tail_obj = elec_tail + gen_tail + shed_tail

        return obj + (self.tail_weight - 1.0) * tail_obj

    def forward(self, input_dict):
        # get target variables
        for comp in self.pre_components:
            input_dict.update(comp(input_dict))
        x_base = input_dict[self.target_key]
        T = self.loss_fn.horizon

        best_x = None
        best_viol_mean = None
        best_obj_mean = None

        for restart_id in range(self.restarts):
            d = 1.0
            prev_viol = None
            non_improve_steps = 0

            if restart_id == 0 or self.restart_noise_std <= 0.0:
                x = x_base.detach().clone()
            else:
                noise = self.restart_noise_std * torch.randn_like(x_base)
                x = x_base.detach().clone() + noise
            x = x.requires_grad_(True)
            input_dict[self.target_key] = x

            # project gradient
            for _ in range(self.max_iters):
                # forward pass in components
                for comp in self.post_components:
                    input_dict.update(comp(input_dict))

                # get corresponding violation
                viol = self.loss_fn.cal_constr_viol(input_dict)
                obj = self._tail_weighted_obj(input_dict)

                # check stopping condition
                current_viol = float(viol.mean().detach().cpu().item())
                if viol.max() < self.viol_tol:
                    break

                if prev_viol is not None:
                    improve = prev_viol - current_viol
                    if improve < self.min_viol_improve:
                        non_improve_steps += 1
                    else:
                        non_improve_steps = 0
                    if non_improve_steps >= self.early_stop_patience:
                        break
                prev_viol = current_viol

                # get gradients
                if self.obj_guard_weight > 0.0:
                    proj_metric = viol.sum() + self.obj_guard_weight * obj.sum()
                else:
                    proj_metric = viol.sum()
                grad = torch.autograd.grad(proj_metric, x, retain_graph=False, create_graph=False)[0]

                # update
                x = (x - d * self.step_size * grad).detach()

                # ENFORCE CH_DIS_MUTEX DIRECTLY (strong projection)
                # x layout (9T): [p_grid_buy(0:T), p_grid_sell(T:2T), p_gen(2T:3T),
                #                 p_ch(3T:4T), p_dis(4T:5T), s_load(5T:6T),
                #                 u_gen(6T:7T), u_ch(7T:8T), u_dis(8T:9T)]
                u_ch_idx = slice(7 * T, 8 * T)
                u_dis_idx = slice(8 * T, 9 * T)

                u_ch = x[:, u_ch_idx]
                u_dis = x[:, u_dis_idx]

                dual_on = (u_ch > 0.5) & (u_dis > 0.5)
                if dual_on.any():
                    zero_ch = (u_ch < u_dis) & dual_on
                    zero_dis = (~zero_ch) & dual_on
                    u_ch = torch.where(zero_ch, 0.1 * u_ch, u_ch)
                    u_dis = torch.where(zero_dis, 0.1 * u_dis, u_dis)
                    x[:, u_ch_idx] = u_ch
                    x[:, u_dis_idx] = u_dis

                d = self.decay * d
                x = x.requires_grad_(True)
                input_dict[self.target_key] = x

            # Final score for this restart: prioritize lower viol then lower obj.
            with torch.no_grad():
                for comp in self.post_components:
                    input_dict.update(comp(input_dict))
                viol_mean = float(self.loss_fn.cal_constr_viol(input_dict).mean().detach().cpu().item())
                obj_mean = float(self.loss_fn.cal_obj(input_dict).mean().detach().cpu().item())

            if (
                best_x is None
                or viol_mean < best_viol_mean - 1e-12
                or (abs(viol_mean - best_viol_mean) <= 1e-12 and obj_mean < best_obj_mean)
            ):
                best_x = x.detach().clone()
                best_viol_mean = viol_mean
                best_obj_mean = obj_mean

        x = best_x
        input_dict[self.target_key] = x
        
        # ============================================================
        # FINAL POWER BALANCE ENFORCEMENT (before returning)
        # ============================================================
        # This ensures that the final solution respects power balance
        # by computing one variable (s_load) to satisfy the constraint.
        load = input_dict[self.loss_fn.load_key]
        pv = input_dict[self.loss_fn.pv_key]
        
        # Unpack x with proper indices for 9T layout
        p_grid_buy_idx = slice(0 * T, 1 * T)
        p_grid_sell_idx = slice(1 * T, 2 * T)
        p_gen_idx = slice(2 * T, 3 * T)
        p_ch_idx = slice(3 * T, 4 * T)
        p_dis_idx = slice(4 * T, 5 * T)
        s_load_idx = slice(5 * T, 6 * T)
        u_gen_idx = slice(6 * T, 7 * T)
        u_ch_idx = slice(7 * T, 8 * T)
        u_dis_idx = slice(8 * T, 9 * T)
        
        p_grid_buy = x[:, p_grid_buy_idx]
        p_grid_sell = x[:, p_grid_sell_idx]
        p_gen = x[:, p_gen_idx]
        p_ch = x[:, p_ch_idx]
        p_dis = x[:, p_dis_idx]
        u_gen = torch.clamp(x[:, u_gen_idx], min=0.0, max=1.0)
        u_ch = (torch.clamp(x[:, u_ch_idx], min=0.0, max=1.0) >= 0.5).float()
        u_dis = (torch.clamp(x[:, u_dis_idx], min=0.0, max=1.0) >= 0.5).float()
        p_grid = p_grid_buy - p_grid_sell

        # Deterministic hard-repair for capacity constraints.
        # This directly addresses recurring dis_cap/ch_cap violations.
        p_gen = torch.clamp(p_gen, min=0.0)
        p_ch = torch.clamp(p_ch, min=0.0)
        p_dis = torch.clamp(p_dis, min=0.0)
        p_gen = torch.minimum(p_gen, self.loss_fn.p_gen_max * u_gen)
        p_ch = torch.minimum(p_ch, self.loss_fn.p_ch_max * u_ch)
        p_dis_cap = self.loss_fn.p_dis_max * u_dis
        if self.end_discharge_steps > 0 and self.end_discharge_scale < 1.0:
            k = min(self.end_discharge_steps, T)
            tail = slice(T - k, T)
            p_dis_cap[:, tail] = p_dis_cap[:, tail] * self.end_discharge_scale
        p_dis = torch.minimum(p_dis, p_dis_cap)

        # Keep at most one battery mode active in hard-repair stage.
        dual_on = (u_ch + u_dis) > 1.0
        keep_ch = u_ch >= u_dis
        u_ch = torch.where(dual_on & keep_ch, torch.ones_like(u_ch), u_ch)
        u_dis = torch.where(dual_on & keep_ch, torch.zeros_like(u_dis), u_dis)
        u_ch = torch.where(dual_on & (~keep_ch), torch.zeros_like(u_ch), u_ch)
        u_dis = torch.where(dual_on & (~keep_ch), torch.ones_like(u_dis), u_dis)

        # Re-apply repaired power caps after possible u_ch/u_dis hard-fix.
        p_ch = torch.minimum(p_ch, self.loss_fn.p_ch_max * u_ch)
        p_dis_cap = self.loss_fn.p_dis_max * u_dis
        if self.end_discharge_steps > 0 and self.end_discharge_scale < 1.0:
            k = min(self.end_discharge_steps, T)
            tail = slice(T - k, T)
            p_dis_cap[:, tail] = p_dis_cap[:, tail] * self.end_discharge_scale
        p_dis = torch.minimum(p_dis, p_dis_cap)
        
        # Power balance: p_gen + p_dis - p_ch + p_grid + pv + s_load = load
        # => s_load = load - p_gen - p_dis + p_ch - p_grid - pv
        s_load_corrected = load - p_gen - p_dis + p_ch - p_grid - pv
        
        # Write repaired variables back to x.
        x[:, p_gen_idx] = p_gen
        x[:, p_ch_idx] = p_ch
        x[:, p_dis_idx] = p_dis
        x[:, u_gen_idx] = u_gen
        x[:, u_ch_idx] = u_ch
        x[:, u_dis_idx] = u_dis

        # Update s_load in x
        x[:, s_load_idx] = s_load_corrected
        
        input_dict[self.target_key] = x
        # Keep downstream evaluation consistent: final repaired candidate should be
        # the one used as policy output, instead of a stale pre-repair x_rnd.
        if "x_rnd" in input_dict:
            input_dict["x_rnd"] = x
        
        return input_dict


if __name__ == "__main__":

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_var = 100
    num_ineq = 100
    hlayers_sol = 5
    hlayers_rnd = 4
    hsize = 256
    batch_size = 64
    lr = 1e-3
    penalty_weight = 100
    num_data = 10000
    test_size = 1000
    val_size = 1000
    train_size = num_data - test_size - val_size

    # init mathmatic model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=60)

    # data sample from uniform distribution
    b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()
    data = {"b":b_samples}
    # data split
    from src.utlis import data_split
    data_train, data_test, data_val = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=False)
    loader_val   = DataLoader(data_val, batch_size, num_workers=0,
                              collate_fn=data_val.collate_fn, shuffle=False)

    # define neural architecture for the solution map smap(p) -> x
    import neuromancer as nm
    from src.func.layer import netFC
    func = netFC(input_dim=num_ineq, hidden_dims=[hsize]*hlayers_sol, output_dim=num_var)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")

    # define rounding model
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    layers_rnd = netFC(input_dim=num_ineq+num_var, hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")

    # build neuromancer components
    components = nn.ModuleList([smap, rnd])

    # build neuromancer problem
    from src.problem import nmQuadratic
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)

    # training
    from src.problem.neuromancer.trainer import trainer
    epochs = 200                    # number of training epochs
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    warmup = 40                     # number of epochs to wait before enacting early stopping policies
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs=epochs,
                         patience=patience, warmup=warmup)
    # training for the rounding problem
    my_trainer.train(loader_train, loader_val)

    # project
    proj = gradientProjection([smap], [rnd], loss_fn, "x")

    # evaluate
    import time
    from tqdm import tqdm
    import pandas as pd
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        proj(datapoints)
        tock = time.time()
        # assign params
        model.set_param_val({"b":b.cpu().numpy()})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(b.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))

from src.problem.math_solver.microgrid import microgrid as msMicrogrid
from src.problem.neuromancer.microgrid import penaltyLoss as nmMicrogridLoss

T = config.horizon
model = msMicrogrid(horizon=T)

smap = nm.system.Node(
    nm.modules.blocks.MLP(insize=param_dim, outsize=model.nx, hsizes=[hsize]*hlayers_sol, nonlin=nn.ReLU,
                          linear_map=nm.slim.maps["linear"], bias=True),
    ["xi"], ["x"], name="smap"
)

rnd = roundGumbelModel(
    layers=layers_rnd, param_keys=["xi"], var_keys=["x"], output_keys=["x_rnd"],
    int_ind=model.int_ind, bin_ind=model.bin_ind, continuous_update=True, name="round"
)

loss_fn = nmMicrogridLoss(
    ["load","pv","price_buy","price_sell","soc0","x_rnd"],
    horizon=T, penalty_weight=config.penalty
)
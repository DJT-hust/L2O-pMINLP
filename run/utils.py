#!/usr/bin/env python
# coding: utf-8
"""
Utlities
"""

import torch
from src.problem.neuromancer.trainer import trainer

def train(components, loss_fn, loader_train, loader_val, lr, penalty_growth,
          patience=20, warmup=None,
          validate_every=125,
          train_eval_batches=8,
          lr_anneal=False, lr_min=1e-6,
          loader_test=None,
          tensorboard=False, tb_logdir=None, tb_run_name="run"):
    epochs = 200                    # number of training epochs
    if penalty_growth:
        growth_rate = 1.03          # growth rate of penalty weight (balance constraint satisfaction and objective)
        default_warmup = 50         # number of validation checks to wait before early stopping
    else:
        growth_rate = 1             # growth rate of penalty weight
        default_warmup = 20         # number of validation checks to wait before early stopping
    if warmup is None:
        warmup = default_warmup
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    scheduler = None
    if lr_anneal:
        total_steps = max(1, epochs * len(loader_train))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=lr_min,
        )
        print(f"[LR] CosineAnnealing enabled: initial_lr={lr}, eta_min={lr_min}, total_steps={total_steps}")
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs=epochs,
                         growth_rate=growth_rate, patience=patience, warmup=warmup,
                         validate_every=validate_every,
                         train_eval_batches=train_eval_batches,
                         scheduler=scheduler,
                         device="cuda", tensorboard=tensorboard,
                         tb_logdir=tb_logdir, tb_run_name=tb_run_name)
    # training for the rounding problem
    my_trainer.train(loader_train, loader_val, loader_test=loader_test)

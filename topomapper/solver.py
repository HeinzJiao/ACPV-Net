#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizer and learning-rate scheduler builders.

This module centralizes optimizer construction and scheduler setup for the
training scripts in this repository.
"""

import torch
import math
from torch.optim.lr_scheduler import LinearLR, LambdaLR, SequentialLR

def make_optimizer(cfg, model):
    """
    Build the optimizer.

    For AdamW, this function follows an MMSeg-style paramwise setup.
    For other optimizers, it keeps the original HiSup-style behavior.
    """
    params = []
    base_lr = cfg.SOLVER.BASE_LR
    backbone_lr_mult = getattr(cfg.SOLVER, 'BACKBONE_LR_MULT', 1.0)  # Do not downscale by default.
    betas = cfg.SOLVER.BETAS
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    # --- Optional LDM-specific LR and weight decay ---
    ldm_lr_enable = bool(getattr(cfg.SOLVER, 'LDM_LR_ENABLE', False))
    ldm_lr = float(getattr(cfg.SOLVER, 'LDM_LR', 2e-6))
    ldm_wd = float(getattr(cfg.SOLVER, 'LDM_WEIGHT_DECAY', 0.0))
    ldm_keys = list(getattr(cfg.SOLVER, 'LDM_PARAM_NAME_KEYWORDS',
                            ["ldm", "diffusion_unet", "denoiser", "model"]))

    def is_ldm_param(name: str) -> bool:
        name_l = name.lower()
        return any(kw.lower() in name_l for kw in ldm_keys)

    if cfg.SOLVER.OPTIMIZER == 'ADAMW':
        # === MMSeg-style AdamW with paramwise behavior ===
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            group_lr = base_lr
            group_wd = weight_decay

            # === Simulate MMSeg paramwise behavior ===
            if "absolute_pos_embed" in name:
                group_wd = 0.0
            elif "relative_position_bias_table" in name:
                group_wd = 0.0
            elif "norm" in name:
                group_wd = 0.0

            if "backbone" in name and backbone_lr_mult != 1.0:
                group_lr = base_lr * backbone_lr_mult

            # === Optional LDM-specific LR / weight decay override ===
            if ldm_lr_enable and is_ldm_param(name):
                group_lr = ldm_lr
                group_wd = ldm_wd

            params.append({"params": [param], "lr": group_lr, "weight_decay": group_wd})

        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            betas=betas,  # Keep the same convention as MMSeg.
            weight_decay=weight_decay
        )

    else:
        # === Original HiSup-style setup ===
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            group_lr = base_lr
            group_wd = weight_decay

            # Optional LDM-specific LR / weight decay override.
            if ldm_lr_enable and is_ldm_param(name):
                group_lr = ldm_lr
                group_wd = ldm_wd

            params.append({"params": [param], "lr": group_lr, "weight_decay": group_wd})

        if cfg.SOLVER.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(
                params,
                lr=base_lr,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=weight_decay
            )

        elif cfg.SOLVER.OPTIMIZER in ['ADAM', 'ADAMcos']:
            optimizer = torch.optim.Adam(
                params,
                lr=base_lr,
                weight_decay=weight_decay,
                amsgrad=cfg.SOLVER.AMSGRAD
            )

        else:
            raise NotImplementedError(f"Unsupported optimizer type: {cfg.SOLVER.OPTIMIZER}")

    return optimizer

def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.OPTIMIZER == 'ADAMcos':
        t = cfg.SOLVER.STATIC_STEP
        max_ep = cfg.SOLVER.MAX_EPOCH
        lambda1 = lambda epoch: 1 if epoch < t else 0.00001 \
            if 0.5*(1+math.cos(math.pi*(epoch-t)/(max_ep-t))) < 0.00001 \
            else 0.5*(1+math.cos(math.pi*(epoch-t)/(max_ep-t)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    elif cfg.SOLVER.LR_SCHEDULER == 'linear_poly':
        # === MMSeg-style linear warmup + PolyLR ===
        warmup_iters = cfg.SOLVER.WARMUP_ITERS  # e.g. 1500
        total_iters = cfg.SOLVER.TOTAL_ITERS  # e.g. 160000
        power = cfg.SOLVER.POLY_POWER  # e.g. 1.0
        eta_min = cfg.SOLVER.ETA_MIN  # e.g. 0.0

        linear_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,
            total_iters=warmup_iters
        )

        base_lr = cfg.SOLVER.BASE_LR  # 6e-5

        def poly_lr_lambda(current_iter):
            if current_iter <= warmup_iters:
                return 1.0  # 1.0 is used by SequentialLR here, not the actual LR value.
            factor = (1 - (current_iter - warmup_iters) / (total_iters - warmup_iters))
            factor = max(factor, 0.0)
            poly = factor ** power
            return (base_lr - eta_min) / base_lr * poly + eta_min / base_lr

        poly_scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, poly_scheduler],
            milestones=[warmup_iters]
        )
        return scheduler

    elif cfg.SOLVER.LR_SCHEDULER == 'warmup_multistep':
        warmup_iters = int(cfg.SOLVER.WARMUP_ITERS)
        milestones_global = list(cfg.SOLVER.STEPS)
        # MultiStepLR restarts counting after warmup, so milestones must be shifted.
        milestones = [max(0, m - warmup_iters) for m in milestones_global]

        linear = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, total_iters=warmup_iters
        )
        multistep = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=cfg.SOLVER.GAMMA
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[linear, multistep], milestones=[warmup_iters]
        )
        return scheduler

    else:
        # Fall back to MultiStepLR by default.
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)

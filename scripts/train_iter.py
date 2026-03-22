#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a detector with iteration-based scheduling.

This script builds the training dataset, resolves the correct detector
implementation from the configuration, and runs the main optimization loop
with optional EMA and checkpoint resume support.
"""

import os
import time
import argparse
import logging
import random
import numpy as np
import datetime
from topomapper.config import cfg
from topomapper.dataset import build_train_dataset
from topomapper.utils.comm import to_single_device
from topomapper.solver import make_lr_scheduler, make_optimizer
from topomapper.utils.logger import setup_logger
from topomapper.utils.miscellaneous import save_config
from topomapper.utils.metric_logger import MetricLogger
from topomapper.utils.checkpoint import DetectronCheckpointer
from collections import Counter
from yacs.config import CfgNode as CN

import torch

torch.multiprocessing.set_sharing_strategy('file_system')


class LossReducer(object):
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        weighted_loss_dict = {}
        for k in self.loss_weights.keys():
            if k in loss_dict:
                weighted_loss_dict[k] = self.loss_weights[k] * loss_dict[k]
            else:
                raise KeyError(f"Loss key '{k}' not found in loss_dict!")

        total_loss = sum(weighted_loss_dict.values())
        return total_loss, weighted_loss_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument("--config-file", metavar="FILE", type=str, default=None)
    parser.add_argument("--clean", default=False, action='store_true')
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint.")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Name of the checkpoint to resume from.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def debug_print_scheduler(scheduler, tag=""):
    import torch
    print(f"\n===== Scheduler Debug {tag} =====")
    print("type:", type(scheduler).__name__)
    print("outer.last_epoch:", getattr(scheduler, "last_epoch", None))
    print("optimizer lrs:", [pg["lr"] for pg in scheduler.optimizer.param_groups])

    if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
        # Internal PyTorch fields.
        print("milestones:", getattr(scheduler, "_milestones", None))
        print("milestone_index:", getattr(scheduler, "_milestone_index", None))
        inner_list = getattr(scheduler, "_schedulers", [])
        for i, s in enumerate(inner_list):
            print(f"  - inner[{i}] type:", type(s).__name__)
            print(f"    inner[{i}].last_epoch:", getattr(s, "last_epoch", None))
            sd = s.state_dict()
            # Print only the most relevant fields.
            keys = ["last_epoch", "gamma", "milestones", "total_iters", "start_factor"]
            print("    inner state:", {k: sd[k] for k in keys if k in sd})
    else:
        sd = scheduler.state_dict()
        print("state_dict keys:", list(sd.keys()))
        print("state_dict:", sd)
    print("===== End Debug =====\n")


def force_align_multistep_lr(scheduler, optimizer, iteration, cfg):
    """
    Force SequentialLR(LinearLR -> MultiStepLR) to align with the global
    iteration count by overwriting the current LR and synchronizing the
    scheduler cache.
    """
    # 1) Access the inner schedulers.
    seq = scheduler  # SequentialLR
    linear, multistep = seq._schedulers  # [LinearLR, MultiStepLR]

    # 2) Overwrite milestones from cfg.
    warm = int(cfg.SOLVER.WARMUP_ITERS)
    steps_after = [max(0, int(s) - warm) for s in cfg.SOLVER.STEPS]
    seq._milestones = [warm]
    multistep.milestones = Counter(steps_after)
    multistep.gamma = float(cfg.SOLVER.GAMMA)

    # 3) Set the internal counters to the current global iteration.
    seq._milestone_index = 1  # Already in the MultiStepLR phase.
    linear.last_epoch = warm  # Warmup is treated as finished.
    multistep.last_epoch = iteration - warm
    seq.last_epoch = iteration

    # 4) Compute the LR for the current global iteration and write it back.
    base_lr = float(cfg.SOLVER.BASE_LR)
    gamma = float(cfg.SOLVER.GAMMA)
    n_decay = sum(iteration >= s for s in cfg.SOLVER.STEPS)
    lr_now = base_lr * (gamma ** n_decay)

    for pg in optimizer.param_groups:
        pg["lr"] = lr_now

    # Synchronize the scheduler cache to avoid stale LR values in the first log.
    seq._last_lr = [lr_now for _ in optimizer.param_groups]
    linear._last_lr = [lr_now for _ in optimizer.param_groups]
    multistep._last_lr = [lr_now for _ in optimizer.param_groups]


def _inner(model):
    # Support both DDP-wrapped and plain single-device models.
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def train(cfg):
    # -------------------- Logger --------------------
    logger = logging.getLogger("training")
    device = cfg.MODEL.DEVICE

    # -------------------- Build model --------------------
    dataset_name = cfg.DATASETS.TRAIN[0]

    # Backbones supported by the unified UPerNet detector branch.
    BACKBONES_FOR_UPER = {"VMamba", "HRNet48v2", "HRNet32v2", "HRNet18v2", "ResNetUNet101"}

    if cfg.MODEL.NAME not in BACKBONES_FOR_UPER:
        raise ValueError(f"Unsupported backbone for UPer detector: {cfg.MODEL.NAME}")

    uc = getattr(cfg.MODEL, "unet_config", None)
    use_ldm = isinstance(uc, CN) and bool(uc.get("target", None))

    enable_mask = bool(getattr(cfg.MODEL, "ENABLE_MASK_BRANCH", True))
    dual_seg = bool(getattr(cfg.MODEL, "DUAL_SEG", False))
    joint_latent = bool(getattr(cfg.MODEL, "JOINT_LATENT", False))

    # Dataset-based task tags.
    is_vh = dataset_name == "deventer_512_train_with_vertex_heatmap"
    is_mask_only = dataset_name == "deventer_512_train"
    is_vh_afm = dataset_name == "deventer_512_train_with_vertex_heatmap_afm"

    # ----------------------------
    # Group A: LDM-based variants
    # ----------------------------
    if use_ldm:
        # A1) Mask + vertex heatmap (LDM): default path.
        if enable_mask:
            if dual_seg:
                if joint_latent:
                    from topomapper.upernet_detector_vh_m_ldm_dualseg_jointlatent import BuildingUPerNetDualSegJointLatentDetector as BuildingUPerNetDetector
                else:
                    from topomapper.upernet_detector_vh_m_ldm_dualseg import BuildingUPerNetDualSegDetector as BuildingUPerNetDetector
            else:
                from topomapper.upernet_detector_vh_m_ldm import BuildingUPerNetDetector
        # A2) Vertex heatmap (LDM) only.
        else:
            from topomapper.upernet_detector_vh_ldm import BuildingUPerNetDetector

    # ----------------------------
    # Group B: Non-LDM variants
    # ----------------------------
    # B1) Vertex heatmap task (no LDM): select decoder family by upsample method.
    elif is_vh:
        upsample_method = cfg.MODEL.DECODE_HEAD.get("UPSAMPLE_METHOD", None)

        if upsample_method in ["deconv", "bilinear", "nearest"]:
            from topomapper.upernet_detector_vertexheatmap_deconv import BuildingUPerNetDetector
        else:
            from topomapper.upernet_detector_vertexheatmap import BuildingUPerNetDetector

    # B2) Mask-only baseline.
    elif is_mask_only:
        from topomapper.upernet_detector import BuildingUPerNetDetector

    # B3) Vertex heatmap + AFM constraint (no LDM).
    elif is_vh_afm:
        from topomapper.upernet_detector_vertexheatmap_afm import BuildingUPerNetDetector

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name} or model configuration: {cfg.MODEL.NAME}")

    model = BuildingUPerNetDetector(cfg)

    model = model.to(device)

    # -------------------- Build training data --------------------
    train_dataset = build_train_dataset(cfg)

    # -------------------- Optimizer and scheduler --------------------
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # -------------------- Loss reducer --------------------
    loss_reducer = LossReducer(cfg)

    # -------------------- Checkpointer --------------------
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)

    # -------------------- Resume mode --------------------
    resume = cfg.get("RESUME", False)
    resume_ckpt_name = cfg.get("RESUME_CHECKPOINT", None)

    if resume and resume_ckpt_name is not None:
        resume_ckpt_path = os.path.join(cfg.OUTPUT_DIR, resume_ckpt_name)
        checkpoint_data = checkpointer.load(resume_ckpt_path)
        iteration = int(checkpoint_data["iteration"])
        logger.info(f"Resume training from checkpoint {resume_ckpt_name}, starting from iteration {iteration}")

        # debug_print_scheduler(scheduler, tag="after load (before any step)")

        scheduler = make_lr_scheduler(cfg, optimizer)  # Rebuild a clean scheduler.

        force_align_multistep_lr(scheduler, optimizer, iteration, cfg)
        logger.info(f"LR aligned: {optimizer.param_groups[0]['lr']:.6e}")

        # current_lr = optimizer.param_groups[0]["lr"]
        # logger.info(f"Current learning rate after resume: {current_lr:.6e}")
    else:
        iteration = 0
        logger.info("Start training from scratch.")

    # -------------------- Main training loop --------------------
    max_iter = cfg.SOLVER.TOTAL_ITERS
    start_training_time = time.time()
    end = time.time()

    meters = MetricLogger(" ")
    model.train()

    use_ema = cfg.get("USE_EMA", False)
    logger.info(f"EMA enabled? model.use_ema={_inner(model).use_ema}")

    while iteration < max_iter:
        # ---------- Per-batch iteration ----------
        for it, (images, annotations) in enumerate(train_dataset):
            if iteration >= max_iter:
                break

            data_time = time.time() - end
            images = images.to(device)
            annotations = to_single_device(annotations, device)

            # Forward pass and loss computation.
            loss_dict, _ = model(images, annotations, iteration, cfg.OUTPUT_DIR)

            total_loss, weighted_loss_dict = loss_reducer(loss_dict)

            # Logging without gradient tracking.
            with torch.no_grad():
                # 1) Weighted losses used for optimization and logging.
                weighted_logs = {k: v.item() for k, v in weighted_loss_dict.items()}

                # 2) Debug metrics prefixed with 'dbg_'.
                debug_logs = {}
                for k, v in loss_dict.items():
                    if k.startswith("dbg_"):
                        # v may be a tensor or a float.
                        debug_logs[k] = (v.detach().item() if torch.is_tensor(v) else float(v))

                # 3) Update meters with both weighted losses and debug metrics.
                meters.update(loss=total_loss, **weighted_logs, **debug_logs)

            # Backpropagation and parameter update.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update EMA after optimizer.step() if enabled.
            if use_ema:
                _inner(model).on_train_batch_end()

            scheduler.step()

            iteration += 1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # Print logs every 20 iterations and at the final iteration.
            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.10f}",
                        "max mem: {memory:.0f}\n",
                    ]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

                print(optimizer.param_groups[-1]["lr"])

                # # Optional: print the learning rate of each parameter.
                # if iteration % 20 == 0:
                #     name_to_lr = {}
                #     for pg in optimizer.param_groups:
                #         for p in pg["params"]:
                #             for name, param in model.named_parameters():
                #                 if param is p:
                #                     name_to_lr[name] = pg["lr"]
                #
                #     logger.info("=== Parameter LRs ===")
                #     for name, lr in name_to_lr.items():
                #         logger.info(f"{name}: {lr:.2e}")

            # ---------- Save checkpoint ----------
            if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                checkpointer.save('model_iter_{:07d}'.format(iteration), iteration=iteration)

    # -------------------- Final training summary --------------------
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / iter)".format(
        total_time_str, total_training_time / max_iter))


if __name__ == "__main__":
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.resume:
        cfg.RESUME = args.resume
    if args.resume_checkpoint:
        cfg.RESUME_CHECKPOINT = args.resume_checkpoint
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        if os.path.isdir(output_dir) and args.clean:
            import shutil

            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('training', output_dir, out_file='train.log')
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))

    save_config(cfg, output_config_path)
    set_random_seed(args.seed, True)
    train(cfg)

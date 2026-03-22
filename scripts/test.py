#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run model inference and evaluation under a given configuration.

This script resolves the correct detector implementation from the config,
loads the latest checkpoint from `cfg.OUTPUT_DIR`, and executes the testing
pipeline.
"""

import os
import argparse
import logging

import torch
from yacs.config import CfgNode as CN

from topomapper.config import cfg
from topomapper.utils.logger import setup_logger
from topomapper.utils.checkpoint import DetectronCheckpointer
from tools.test_pipelines import TestPipeline

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default=None,
                        )

    parser.add_argument("--eval-type",
                        type=str,
                        help="Evalutation type for the test results",
                        default="coco_iou",
                        choices=["coco_iou",  "boundary_iou", "polis"]
                        )

    parser.add_argument("--sampler",
                        type=str,
                        help="Sampling method to use",
                        default="direct",
                        choices=["direct", "ddim", "ddpm"]
                        )

    parser.add_argument("--ddim-steps",
                        type=int,
                        help="Number of DDIM sampling steps",
                        default=200)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    return parser.parse_args()


def resolve_model_class(cfg):
    """
    Resolve which model implementation to use.

    Design overview:
    - Default path:
      * LDM enabled + mask branch enabled
      * mask predicted by UPerNet (VMamba/HRNet/etc.), vertex heatmap reconstructed by LDM conditioned on UPerNet features
        -> upernet_detector_vh_m_ldm.BuildingUPerNetDetector

    - Other supported paths:
      1) Vertex heatmap only (no mask):
         a) with LDM  -> upernet_detector_vh_ldm
         b) without LDM:
            - heatmap decoder variants (deconv/bilinear/nearest) -> upernet_detector_vertexheatmap_deconv
            - default heatmap decoder -> upernet_detector_vertexheatmap
            - + AFM constraint -> upernet_detector_vertexheatmap_afm

      2) Mask-only baseline (no vertex heatmap):
         -> upernet_detector
    """
    dataset_name = cfg.DATASETS.TRAIN[0]

    # Backbones supported by the unified UPerNet detector branch.
    BACKBONES_FOR_UPER = {"VMamba", "HRNet48v2", "HRNet32v2", "HRNet18v2", "ResNetUNet101"}
    if cfg.MODEL.NAME not in BACKBONES_FOR_UPER:
        raise ValueError(f"Unsupported backbone for UPer detector: {cfg.MODEL.NAME}")

    # LDM is enabled when cfg.MODEL.unet_config.target exists and is non-empty.
    uc = getattr(cfg.MODEL, "unet_config", None)
    use_ldm = isinstance(uc, CN) and bool(uc.get("target", None))

    # Mask branch toggle.
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
            return BuildingUPerNetDetector

        # A2) Vertex heatmap (LDM) only.
        from topomapper.upernet_detector_vh_ldm import BuildingUPerNetDetector
        return BuildingUPerNetDetector

    # ----------------------------
    # Group B: Non-LDM variants
    # ----------------------------
    # B1) Vertex heatmap task (no LDM): select decoder family by upsample method.
    if is_vh:
        upsample_method = cfg.MODEL.DECODE_HEAD.get("UPSAMPLE_METHOD", None)
        if upsample_method in {"deconv", "bilinear", "nearest"}:
            from topomapper.upernet_detector_vertexheatmap_deconv import BuildingUPerNetDetector
            return BuildingUPerNetDetector

        from topomapper.upernet_detector_vertexheatmap import BuildingUPerNetDetector
        return BuildingUPerNetDetector

    # B2) Mask-only baseline
    if is_mask_only:
        from topomapper.upernet_detector import BuildingUPerNetDetector
        return BuildingUPerNetDetector

    # B3) Vertex heatmap + AFM constraint (no LDM).
    if is_vh_afm:
        from topomapper.upernet_detector_vertexheatmap_afm import BuildingUPerNetDetector
        return BuildingUPerNetDetector

    raise ValueError(
        f"Unsupported dataset '{dataset_name}' or incompatible model configuration '{cfg.MODEL.NAME}'."
    )


def test(cfg, args):
    logger = logging.getLogger("testing")
    device = cfg.MODEL.DEVICE

    # Build the model from the configuration.
    ModelCls = resolve_model_class(cfg)
    model = ModelCls(cfg).to(device)

    # Log EMA usage if enabled.
    inner = model.module if hasattr(model, "module") else model
    if getattr(inner, "use_ema", False):
        logger.info(f"EMA enabled for inference (decay={getattr(inner, 'ema_decay', None)}).")

    # Load the checkpoint.
    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
        _ = checkpointer.load()        
        model = model.eval()

    # Run the testing pipeline.
    logger.info(f"Checkpoint load dir (cfg.OUTPUT_DIR): {cfg.OUTPUT_DIR}")
    logger.info(f"Test output dir: {TestPipeline(cfg, args.eval_type).output_dir}")
    test_pipeline = TestPipeline(cfg, args.eval_type)
    test_pipeline.test(model)
    # test_pipeline.eval()


if __name__ == "__main__":
    args = parse_args()

    # Merge the config file if provided.
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)

    # Inject sampler settings from the command line into cfg.
    if args.sampler is not None:
        cfg.SAMPLER = args.sampler
    if args.ddim_steps is not None:
        cfg.DDIM_STEPS = args.ddim_steps

    # Merge additional command-line overrides and freeze the config.
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('testing', output_dir)

    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")

    test(cfg, args)

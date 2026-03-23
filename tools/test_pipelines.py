#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing pipelines for model inference output export.

This module runs dataset-level inference and saves masks, probability maps,
and optional latent reconstructions for downstream evaluation and analysis.
"""

import os
import os.path as osp
import json
import torch
import logging
import numpy as np
import scipy
from PIL import Image
from tqdm import tqdm
from skimage import io
from topomapper.utils.comm import to_single_device
from topomapper.dataset import build_test_dataset

# Class ID -> RGB color mapping.
id_to_color = {
        4: (120, 174, 237),  # Water      
        3: (173, 226, 181),  # Vegetation  
        2: (251, 245, 234),  # Unvegetated 
        1: (141, 167, 193),  # Road        
        0: (242, 224, 199),  # Building   
    }

# Binary visualization color mapping for background / foreground outputs.
binary_id_to_color = {
    0: (0, 0, 0),        # background
    1: (255, 255, 255),  # foreground
}

class TestPipeline():
    def __init__(self, cfg, eval_type='coco_iou'):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.eval_type = eval_type
        self.use_ldm = cfg.MODEL.get("unet_config", None)
        self.sampler = cfg.get("SAMPLER", None)

        # Default: cfg.OUTPUT_DIR (keep old behavior)
        # Override: cfg.TEST.OUTPUT_DIR (only for test-time outputs, e.g., zero-shot)
        self.output_dir = cfg.TEST_OUTPUT_DIR if cfg.TEST_OUTPUT_DIR else cfg.OUTPUT_DIR

        self.gt_file = ''
        self.dt_file = ''
    
    def test(self, model):
        if self.use_ldm:
            self.test_on_deventer_ldm(model, self.dataset_name)
        elif 'deventer' in self.dataset_name:
            self.test_on_deventer(model, self.dataset_name)

    def test_on_deventer(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info(f'Testing on {dataset_name} dataset')

        # Build the test dataset.
        test_dataset = build_test_dataset(self.cfg)

        # Create output directories.
        seg_mask_npy_dir = osp.join(self.output_dir, 'seg_mask_npy')  # Segmentation masks (.npy).
        seg_prob_npy_dir = osp.join(self.output_dir, 'seg_prob_npy')  # Segmentation probabilities (.npy).
        junc_prob_npy_dir = osp.join(self.output_dir, 'junction_prob_npy')  # Junction probability maps (.npy).
        seg_mask_viz_dir = osp.join(self.output_dir, 'seg_mask_viz')  # Segmentation visualizations (.png).

        os.makedirs(seg_mask_npy_dir, exist_ok=True)
        os.makedirs(seg_prob_npy_dir, exist_ok=True)
        os.makedirs(junc_prob_npy_dir, exist_ok=True)
        os.makedirs(seg_mask_viz_dir, exist_ok=True)

        # Run inference.
        logging.getLogger("testing").info(f"Test outputs will be saved to: {self.output_dir}")
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output, 'cpu')

            batch_mask_probs = output['mask_prob']  # (B, num_classes, H, W), per-class probability map.
            batch_seg_masks = output['seg_mask']  # (B, H, W), int64 segmentation label map.
            batch_junc_probs = output['jloc_prob']  # (B, H, W), float junction probability map.
            if 'reseg_mask' in output:
                batch_reseg_masks = output['reseg_mask']

            for b in range(images.size(0)):
                filename = annotations[b]['filename']
                name_no_ext = os.path.splitext(filename)[0]

                seg_mask = batch_seg_masks[b].numpy()  # (H, W)
                seg_prob = batch_mask_probs[b].numpy()  # (num_classes, H, W)
                junc_prob = batch_junc_probs[b].numpy()  # (H, W)

                # Save the segmentation mask (.npy).
                np.save(osp.join(seg_mask_npy_dir, f'{name_no_ext}.npy'), seg_mask)

                # Save the segmentation probability map (.npy).
                np.save(osp.join(seg_prob_npy_dir, f'{name_no_ext}.npy'), seg_prob)

                # Save the junction probability map (.npy).
                np.save(osp.join(junc_prob_npy_dir, f'{name_no_ext}.npy'), junc_prob)

                # Visualize the segmentation mask.
                h, w = seg_mask.shape
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                for class_id, color in id_to_color.items():
                    color_mask[seg_mask == class_id] = color

                Image.fromarray(color_mask).save(osp.join(seg_mask_viz_dir, f'{name_no_ext}.png'))

        logger.info(f'All segmentation masks (.npy) saved to: {seg_mask_npy_dir}')
        logger.info(f'All segmentation probs (.npy) saved to: {seg_prob_npy_dir}')
        logger.info(f'All segmentation visualizations saved to: {seg_mask_viz_dir}')
        logger.info(f'All junction probability maps (.npy) saved to: {junc_prob_npy_dir}')

    def test_on_deventer_ldm(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info(f'Testing on {dataset_name} dataset')

        # Build the test dataset.
        test_dataset = build_test_dataset(self.cfg)

        # Create output directories.
        prob_map_npy_dir = osp.join(self.output_dir, 'prob_map_npy')      # probability map (.npy)
        prob_map_viz_dir = osp.join(self.output_dir, 'prob_map_viz')      # Grayscale probability maps (.png).

        seg_mask_npy_dir = osp.join(self.output_dir, 'seg_mask_npy')
        seg_mask_viz_dir = osp.join(self.output_dir, 'seg_mask_viz')
        # Save optional parcel outputs for dual-seg models. Older models do not
        # provide these keys and will skip this block automatically.
        parcel_mask_npy_dir = osp.join(self.output_dir, 'parcel_mask_npy')
        parcel_mask_viz_dir = osp.join(self.output_dir, 'parcel_mask_viz')
        parcel_prob_npy_dir = osp.join(self.output_dir, 'parcel_prob_npy')
        vertex_heatmap_latent_dir = osp.join(self.output_dir, self.sampler, 'vertex_heatmap_latent_pt')
        edge_mask_latent_dir = osp.join(self.output_dir, self.sampler, 'edge_mask_latent_pt')

        os.makedirs(prob_map_npy_dir, exist_ok=True)
        os.makedirs(prob_map_viz_dir, exist_ok=True)
        os.makedirs(seg_mask_npy_dir, exist_ok=True)
        os.makedirs(seg_mask_viz_dir, exist_ok=True)
        os.makedirs(vertex_heatmap_latent_dir, exist_ok=True)

        # Run inference.
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output, 'cpu')

            batch_mask_probs = output['mask_prob']  # (B, num_classes, H, W), per-class probability map.
            batch_seg_masks = output['seg_mask']  # (B, H, W), int64 segmentation label map.
            # Select binary or multi-class visualization colors from the output channel count.
            seg_id_to_color = binary_id_to_color if batch_mask_probs.shape[1] == 2 else id_to_color

            # Optional parcel outputs for dual-seg models; None for older models.
            batch_parcel_probs = output.get('parcel_mask_prob', None)
            batch_parcel_masks = output.get('parcel_seg_mask', None)
            vertex_heatmap_latent_recon = output['vertex_heatmap_latent_recon']  # shape: (B, 3, H/4, W/4)
            edge_mask_latent_recon = output.get('edge_mask_latent_recon', None)

            for b in range(images.size(0)):
                filename = annotations[b]['filename']
                name_no_ext = os.path.splitext(filename)[0]

                # Save the segmentation mask (.npy).
                seg_mask = batch_seg_masks[b].numpy()  # (H, W)
                np.save(osp.join(seg_mask_npy_dir, f'{name_no_ext}.npy'), seg_mask)

                # Visualize the segmentation mask.
                h, w = seg_mask.shape
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                for class_id, color in seg_id_to_color.items():
                    color_mask[seg_mask == class_id] = color
                Image.fromarray(color_mask).save(osp.join(seg_mask_viz_dir, f'{name_no_ext}.png'))

                # Save the probability map (.npy).
                prob_map = batch_mask_probs[b].numpy().astype(np.float32)  # (C, H, W)
                np.save(osp.join(prob_map_npy_dir, f'{name_no_ext}.npy'), prob_map)

                # If parcel outputs are available, also save parcel masks,
                # probability maps, and visualizations.
                if batch_parcel_masks is not None:
                    os.makedirs(parcel_mask_npy_dir, exist_ok=True)
                    os.makedirs(parcel_mask_viz_dir, exist_ok=True)

                    parcel_mask = batch_parcel_masks[b].numpy()
                    np.save(osp.join(parcel_mask_npy_dir, f'{name_no_ext}.npy'), parcel_mask)

                    ph, pw = parcel_mask.shape
                    parcel_color_mask = np.zeros((ph, pw, 3), dtype=np.uint8)
                    for class_id, color in binary_id_to_color.items():
                        parcel_color_mask[parcel_mask == class_id] = color
                    Image.fromarray(parcel_color_mask).save(osp.join(parcel_mask_viz_dir, f'{name_no_ext}.png'))

                    if batch_parcel_probs is not None:
                        os.makedirs(parcel_prob_npy_dir, exist_ok=True)
                        parcel_prob_map = batch_parcel_probs[b].numpy().astype(np.float32)
                        np.save(osp.join(parcel_prob_npy_dir, f'{name_no_ext}.npy'), parcel_prob_map)

                # Visualize each probability channel as a grayscale map.
                C, H, W = prob_map.shape
                for class_id in range(C):
                    prob_img = (prob_map[class_id] * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(prob_img, mode='L').save(
                        osp.join(prob_map_viz_dir, f'{name_no_ext}_prob_{class_id}.png')
                    )

                # Save the reconstructed vertex heatmap latent as `.pt`.
                save_path = osp.join(vertex_heatmap_latent_dir, f"{name_no_ext}.pt")
                torch.save(vertex_heatmap_latent_recon[b], save_path)

                if edge_mask_latent_recon is not None:
                    os.makedirs(edge_mask_latent_dir, exist_ok=True)
                    edge_save_path = osp.join(edge_mask_latent_dir, f"{name_no_ext}.pt")
                    torch.save(edge_mask_latent_recon[b], edge_save_path)

        logger.info(f'All vertex heatmap latents (.pt) saved to: {vertex_heatmap_latent_dir}')
        if os.path.isdir(edge_mask_latent_dir):
            logger.info(f'All edge mask latents (.pt) saved to: {edge_mask_latent_dir}')
        # These directories are populated only when parcel outputs are present.
        if os.path.isdir(parcel_mask_npy_dir):
            logger.info(f'Parcel masks (.npy) will be saved to: {parcel_mask_npy_dir}')
        if os.path.isdir(parcel_mask_viz_dir):
            logger.info(f'Parcel mask visualizations will be saved to: {parcel_mask_viz_dir}')
        if os.path.isdir(parcel_prob_npy_dir):
            logger.info(f'Parcel probability maps (.npy) will be saved to: {parcel_prob_npy_dir}')


            

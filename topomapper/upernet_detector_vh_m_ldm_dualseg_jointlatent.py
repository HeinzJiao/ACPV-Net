#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-segmentation + joint-latent diffusion variant.

This model extends `upernet_detector_vh_m_ldm_dualseg.py` by replacing the
vertex-only latent diffusion target with a joint latent composed of:
- edge-mask latent
- vertex-heatmap latent
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from .upernet_detector_vh_m_ldm import visualize
from .upernet_detector_vh_m_ldm import DiffusionWrapper
from .upernet_detector_vh_m_ldm_dualseg import BuildingUPerNetDualSegDetector
from .modules.ema import LitEma


class BuildingUPerNetDualSegJointLatentDetector(BuildingUPerNetDualSegDetector):
    """
    Dual-seg + joint-latent LDM variant.

    Relative to upernet_detector_vh_m_ldm_dualseg.py, this model keeps:
      1) edge segmentation branch
      2) parcel segmentation branch

    and changes the LDM target from:
      - vertex latent only

    to a joint latent:
      - [edge_mask_latent, vertex_heatmap_latent] concatenated on channel dim

    The denoised latent is split back into:
      - edge_mask_latent_recon
      - vertex_heatmap_latent_recon
    """

    def __init__(self, cfg, test=False):
        super().__init__(cfg, test=test)
        if not test:
            self.formatter = DualSegJointLatentAnnotationFormatter(cfg)

        loss_cfg = cfg.MODEL.LOSS

        # Config keys controlling the joint latent diffusion setup.
        self.edge_latent_channels = int(cfg.MODEL.get("EDGE_LATENT_CHANNELS", 3))
        self.vertex_latent_channels = int(cfg.MODEL.get("VERTEX_LATENT_CHANNELS", 3))
        self.loss_edge_latent = float(loss_cfg.get("LOSS_EDGE_LATENT", 1.0))
        self.loss_vertex_latent = float(loss_cfg.get("LOSS_VERTEX_LATENT", 1.0))
        self.joint_latent_channels = self.edge_latent_channels + self.vertex_latent_channels

        # Rebuild the diffusion wrapper so latent_conv matches the joint latent channels.
        self.model = DiffusionWrapper(
            diff_model_config=cfg.MODEL.unet_config,
            conditioning_key=cfg.MODEL.conditioning_key,
            in_channels=self.joint_latent_channels,
            out_channels=cfg.MODEL.unet_config.params.model_channels,
        )
        if self.use_ema:
            self.model_ema = LitEma(
                self.model,
                decay=self.ema_decay,
                use_num_upates=self.ema_use_num_updates,
            )

    def split_joint_latent(self, latent):
        edge_latent = latent[:, :self.edge_latent_channels]
        vertex_latent = latent[:, self.edge_latent_channels:self.edge_latent_channels + self.vertex_latent_channels]
        return edge_latent, vertex_latent

    def build_joint_latent(self, targets):
        edge_latent_gt = targets['edge_mask_latent']
        vertex_latent_gt = targets['vertex_heatmap_latent']
        joint_latent_gt = torch.cat([edge_latent_gt, vertex_latent_gt], dim=1)
        return joint_latent_gt, edge_latent_gt, vertex_latent_gt

    def predict_joint_latent_noise(self, joint_latent_gt, cond, t):
        noise = torch.randn_like(joint_latent_gt)
        latent_noisy = self.q_sample(x_start=joint_latent_gt, t=t, noise=noise)
        pred_noise = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
        return latent_noisy, pred_noise, noise

    def compute_joint_latent_loss(self, pred_noise, noise):
        pred_edge_noise, pred_vertex_noise = self.split_joint_latent(pred_noise)
        gt_edge_noise, gt_vertex_noise = self.split_joint_latent(noise)

        loss_edge = F.l1_loss(pred_edge_noise, gt_edge_noise, reduction='none').mean(dim=(1, 2, 3)).mean()
        loss_vertex = F.l1_loss(pred_vertex_noise, gt_vertex_noise, reduction='none').mean(dim=(1, 2, 3)).mean()

        return {
            'loss_edge_latent_l1': loss_edge * self.loss_edge_latent,
            'loss_vertex_heatmap_l1': loss_vertex * self.loss_vertex_latent,
        }

    def forward_train(self, images, annotations=None, iteration=None, output_dir=None):
        self.train_step += 1

        inputs = self.backbone(images)
        features = self.uper_head(inputs)
        features_aux = self.auxiliary_head(inputs)

        edge_feature = features
        edge_pred = self.mask_predictor(edge_feature)
        edge_pred = F.interpolate(edge_pred, scale_factor=4, mode='bilinear', align_corners=False)
        edge_pred_aux = self.mask_aux_predictor(features_aux)

        parcel_feature = self.parcel_mask_head(features)
        parcel_pred = self.parcel_mask_predictor(parcel_feature)
        parcel_pred = F.interpolate(parcel_pred, scale_factor=4, mode='bilinear', align_corners=False)
        parcel_pred_aux = self.parcel_aux_predictor(features_aux)

        cond = features
        targets, metas = self.formatter(annotations)
        joint_latent_gt, edge_latent_gt, vertex_latent_gt = self.build_joint_latent(targets)
        bsz = joint_latent_gt.shape[0]
        t = torch.randint(0, self.num_timesteps, (bsz,), device=joint_latent_gt.device).long()
        latent_noisy, pred_noise, noise = self.predict_joint_latent_noise(joint_latent_gt, cond, t)

        loss_dict = self.compute_joint_latent_loss(pred_noise, noise)

        edge_gt = targets['mask'].squeeze(1).long()
        loss_dict['loss_mask'] = self.segmentation_loss(
            edge_pred, edge_gt, loss_type=self.mask_loss_type
        )
        loss_dict['loss_mask_aux'] = self.segmentation_loss(
            edge_pred_aux, edge_gt, loss_type=self.mask_aux_loss_type
        )

        parcel_gt = targets['parcel_mask'].squeeze(1).long()
        loss_dict['loss_parcel_mask'] = self.segmentation_loss(
            parcel_pred,
            parcel_gt,
            loss_type=self.parcel_loss_type,
            w_bg=self.parcel_w_bg,
            w_obj=self.parcel_w_obj,
        )
        loss_dict['loss_parcel_mask_aux'] = self.segmentation_loss(
            parcel_pred_aux,
            parcel_gt,
            loss_type=self.parcel_aux_loss_type,
            w_bg=self.parcel_w_bg,
            w_obj=self.parcel_w_obj,
        )

        if iteration is not None and iteration % 10000 == 0:
            with torch.no_grad():
                if getattr(self, "use_ema", False):
                    with self.ema_scope(context=f"viz@iter{iteration}"):
                        pred_noise_ema = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
                        latent_recon = self.predict_start_from_noise(latent_noisy, t, pred_noise_ema)
                else:
                    pred_noise_cur = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
                    latent_recon = self.predict_start_from_noise(latent_noisy, t, pred_noise_cur)

                edge_latent_recon, vertex_latent_recon = self.split_joint_latent(latent_recon)
                joint_edge_visualize(
                    output_dir=output_dir,
                    edge_latent_gt=edge_latent_gt,
                    edge_latent_recon=edge_latent_recon,
                    vertex_latent_gt=vertex_latent_gt,
                    vertex_latent_recon=vertex_latent_recon,
                    latent_noisy=latent_noisy,
                    edge_mask_pred=edge_pred,
                    edge_mask_pred_aux=edge_pred_aux,
                    edge_gt_mask=edge_gt,
                    parcel_mask_pred=parcel_pred,
                    parcel_mask_pred_aux=parcel_pred_aux,
                    parcel_gt_mask=parcel_gt,
                    images=images,
                    iteration=iteration,
                    t=t,
                )

        extra_info = {}
        return loss_dict, extra_info

    def forward_test(self, images, annotations=None, sampler="direct", ddim_steps=200):
        inputs = self.backbone(images)
        features = self.uper_head(inputs)
        features_aux = self.auxiliary_head(inputs)

        edge_feature = features
        edge_pred = self.mask_predictor(edge_feature)
        edge_pred = F.interpolate(edge_pred, scale_factor=4, mode='bilinear', align_corners=False)
        edge_mask_prob = edge_pred.softmax(dim=1)
        edge_seg_mask = edge_mask_prob.argmax(dim=1)

        parcel_feature = self.parcel_mask_head(features)
        parcel_pred = self.parcel_mask_predictor(parcel_feature)
        parcel_pred = F.interpolate(parcel_pred, scale_factor=4, mode='bilinear', align_corners=False)
        parcel_mask_prob = parcel_pred.softmax(dim=1)
        parcel_seg_mask = parcel_mask_prob.argmax(dim=1)

        cond = features
        bsz, _, h4, w4 = features.shape
        latent_shape = (bsz, self.edge_latent_channels + self.vertex_latent_channels, h4, w4)

        if getattr(self, "use_ema", False):
            scope = self.ema_scope(context="inference")
        else:
            from contextlib import contextmanager

            @contextmanager
            def _null():
                yield
            scope = _null()

        with scope:
            if sampler == "direct":
                noise_x = torch.randn(latent_shape, device=cond.device)
                final_t = torch.tensor([self.num_timesteps - 1], device=cond.device).long()
                pred_noise = self.apply_model(x_noisy=noise_x, t=final_t, cond=cond)
                latent_recon = self.predict_start_from_noise(noise_x, final_t, pred_noise)
            elif sampler == "ddim":
                samples, intermediates = self.sample_log(cond=cond, shape=latent_shape, ddim_steps=ddim_steps)
                latent_recon = samples
            else:
                raise ValueError(f"Unsupported sampler: {sampler}")

        edge_latent_recon, vertex_latent_recon = self.split_joint_latent(latent_recon)
        output = {
            "edge_mask_latent_recon": edge_latent_recon,
            "vertex_heatmap_latent_recon": vertex_latent_recon,
            "joint_latent_recon": latent_recon,

            "mask_prob": edge_mask_prob,
            "seg_mask": edge_seg_mask,
            "edge_mask_prob": edge_mask_prob,
            "edge_seg_mask": edge_seg_mask,

            "parcel_mask_prob": parcel_mask_prob,
            "parcel_seg_mask": parcel_seg_mask,
        }
        extra_info = {}
        return output, extra_info


def joint_edge_visualize(output_dir, edge_latent_gt, edge_latent_recon,
                         vertex_latent_gt, vertex_latent_recon, latent_noisy,
                         edge_mask_pred, edge_mask_pred_aux, edge_gt_mask,
                         parcel_mask_pred, parcel_mask_pred_aux, parcel_gt_mask,
                         images, iteration, t):
    # Reuse the parent visualization helper for the vertex latent and edge branch.
    visualize(
        output_dir=output_dir,
        latent_noisy=latent_noisy[:, -vertex_latent_gt.shape[1]:],
        latent_gt=vertex_latent_gt,
        latent_recon=vertex_latent_recon,
        mask_pred=edge_mask_pred,
        mask_pred_aux=edge_mask_pred_aux,
        gt_mask=edge_gt_mask,
        images=images,
        iteration=iteration,
        t=t,
    )

    latent_dir = os.path.join(output_dir, "edge_latent")
    parcel_mask_dir = os.path.join(output_dir, "parcel_mask")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(parcel_mask_dir, exist_ok=True)

    edge_gt_np = edge_latent_gt[0].detach().cpu().numpy()
    edge_recon_np = edge_latent_recon[0].detach().cpu().numpy()

    for c in range(edge_gt_np.shape[0]):
        def to_uint8(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min < 1e-5:
                return np.zeros_like(arr, dtype=np.uint8)
            arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
            return arr.astype(np.uint8)

        gt_vis = cv2.applyColorMap(to_uint8(edge_gt_np[c]), cv2.COLORMAP_VIRIDIS)
        recon_vis = cv2.applyColorMap(to_uint8(edge_recon_np[c]), cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(latent_dir, f'edge_latent_gt_{iteration}_ch{c}.png'), gt_vis)
        cv2.imwrite(os.path.join(latent_dir, f'edge_latent_recon_{iteration}_ch{c}.png'), recon_vis)

    parcel_pred_prob = F.softmax(parcel_mask_pred, dim=1)
    parcel_seg_pred = parcel_pred_prob.argmax(dim=1)[0].detach().cpu().numpy()
    parcel_aux_prob = F.softmax(parcel_mask_pred_aux, dim=1)
    parcel_seg_pred_aux = parcel_aux_prob.argmax(dim=1)[0].detach().cpu().numpy()
    parcel_seg_gt = parcel_gt_mask[0].detach().cpu().numpy()

    def mask_to_rgb(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[mask == 1] = (255, 255, 255)
        return rgb

    cv2.imwrite(os.path.join(parcel_mask_dir, f"parcel_mask_pred_{iteration}.png"), mask_to_rgb(parcel_seg_pred))
    cv2.imwrite(os.path.join(parcel_mask_dir, f"parcel_mask_pred_aux_{iteration}.png"), mask_to_rgb(parcel_seg_pred_aux))
    cv2.imwrite(os.path.join(parcel_mask_dir, f"parcel_mask_gt_{iteration}.png"), mask_to_rgb(parcel_seg_gt))

    print(f"[Visualize] Saved joint-latent dualseg visualization for iteration {iteration} to: {output_dir}")


class DualSegJointLatentAnnotationFormatter(object):
    def __init__(self, cfg):
        self.target_h = cfg.DATASETS.TARGET.HEIGHT
        self.target_w = cfg.DATASETS.TARGET.WIDTH
        self.origin_h = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_w = cfg.DATASETS.ORIGIN.WIDTH

    def __call__(self, annotations):
        targets = []
        metas = []
        for ann in annotations:
            t, m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        return default_collate(targets), metas

    def _process_per_image(self, ann):
        target = {
            'vertex_heatmap_latent': ann['vertex_heatmap_latent'],
            'edge_mask_latent': ann['edge_mask_latent'],
            'mask': ann['mask'],
            'parcel_mask': ann['parcel_mask'],
        }
        meta = {}
        return target, meta

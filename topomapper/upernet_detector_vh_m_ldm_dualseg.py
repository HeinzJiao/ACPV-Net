#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-segmentation variant of the UPerNet + LDM detector.

Relative to `upernet_detector_vh_m_ldm.py`, this model adds a parcel
segmentation branch while keeping the original edge segmentation branch and
the latent vertex heatmap diffusion branch.
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data.dataloader import default_collate

from .upernet_detector_vh_m_ldm import BuildingUPerNetDetector, visualize


class BuildingUPerNetDualSegDetector(BuildingUPerNetDetector):
    """
    Dual-seg variant of BuildingUPerNetDetector.

    Relative to upernet_detector_vh_m_ldm.py, this model adds:
      1) A new parcel segmentation branch (main + aux)
      2) Parcel segmentation losses computed via self.segmentation_loss(...)
      3) Parcel predictions in forward_test outputs

    Existing mask branch from the parent class is treated as the edge branch.

    Reserved cfg keys for the new branch:
      MODEL.LOSS.PARCEL_LOSS_TYPE
      MODEL.LOSS.PARCEL_AUX_LOSS_TYPE
      MODEL.LOSS.PARCEL_W_BG
      MODEL.LOSS.PARCEL_W_OBJ
    """

    def __init__(self, cfg, test=False):
        super().__init__(cfg, test=test)
        if not test:
            self.formatter = DualSegAnnotationFormatter(cfg)

        aux_cfg = cfg.MODEL.AUX_HEAD
        loss_cfg = cfg.MODEL.LOSS

        # Dedicated parcel segmentation branch on top of the shared UPer features.
        self.parcel_mask_head = self._make_conv(self.channels, self.channels, self.channels)
        self.parcel_mask_predictor = self._make_predictor(self.channels, self.num_classes)

        # Dedicated parcel auxiliary predictor mirroring the existing aux branch.
        self.parcel_aux_predictor = nn.Sequential(
            self.dropout,
            nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(aux_cfg.CHANNELS, aux_cfg.NUM_CLASSES, kernel_size=3, padding=1)
        )

        # Config placeholders controlling parcel branch loss behavior.
        self.parcel_loss_type = loss_cfg.get("PARCEL_LOSS_TYPE", self.mask_loss_type).lower()
        self.parcel_aux_loss_type = loss_cfg.get("PARCEL_AUX_LOSS_TYPE", self.mask_aux_loss_type).lower()
        self.parcel_w_bg = float(loss_cfg.get("PARCEL_W_BG", self.w_bg))
        self.parcel_w_obj = float(loss_cfg.get("PARCEL_W_OBJ", self.w_obj))

    def segmentation_loss(self, logits, target, loss_type="ce", w_bg=None, w_obj=None):
        """
        Extension of the parent loss so different branches can use different
        class weights.
        """
        loss_type = loss_type.lower()

        if loss_type == "ce":
            return F.cross_entropy(logits, target)

        elif loss_type == "weighted_ce":
            class_weight = torch.tensor(
                [self.w_bg if w_bg is None else w_bg, self.w_obj if w_obj is None else w_obj],
                dtype=logits.dtype,
                device=logits.device
            )
            return F.cross_entropy(logits, target, weight=class_weight)

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward_train(self, images, annotations=None, iteration=None, output_dir=None):
        self.train_step += 1

        # Shared backbone and shared UPer features.
        inputs = self.backbone(images)
        features = self.uper_head(inputs)
        features_aux = self.auxiliary_head(inputs)

        # Existing edge segmentation branch inherited from the parent model.
        edge_feature = features
        edge_pred = self.mask_predictor(edge_feature)
        edge_pred = F.interpolate(edge_pred, scale_factor=4, mode='bilinear', align_corners=False)
        edge_pred_aux = self.mask_aux_predictor(features_aux)

        # Parcel segmentation branch.
        parcel_feature = self.parcel_mask_head(features)
        parcel_pred = self.parcel_mask_predictor(parcel_feature)
        parcel_pred = F.interpolate(parcel_pred, scale_factor=4, mode='bilinear', align_corners=False)
        parcel_pred_aux = self.parcel_aux_predictor(features_aux)

        # The LDM branch remains conditioned on the shared UPer features.
        cond = features
        targets, metas = self.formatter(annotations)
        latent_gt = targets['vertex_heatmap_latent']
        bsz = latent_gt.shape[0]
        t = torch.randint(0, self.num_timesteps, (bsz,), device=latent_gt.device).long()
        latent_noisy, pred_noise, noise = self.predict_vertex_latent_heatmap_noise(latent_gt, cond, t)

        loss_dict = self.compute_vertex_heatmap_loss(pred_noise, noise)

        # Edge targets reuse the original 'mask' key for backward compatibility.
        edge_gt = targets['mask'].squeeze(1).long()
        loss_dict['loss_mask'] = self.segmentation_loss(
            edge_pred, edge_gt, loss_type=self.mask_loss_type
        )
        loss_dict['loss_mask_aux'] = self.segmentation_loss(
            edge_pred_aux, edge_gt, loss_type=self.mask_aux_loss_type
        )

        # Parcel targets are expected from the dual-seg dataset definition.
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

        if iteration is not None and iteration % 5000 == 0:
            with torch.no_grad():
                if getattr(self, "use_ema", False):
                    with self.ema_scope(context=f"viz@iter{iteration}"):
                        pred_noise_ema = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
                        latent_recon = self.predict_start_from_noise(latent_noisy, t, pred_noise_ema)
                else:
                    pred_noise_cur = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
                    latent_recon = self.predict_start_from_noise(latent_noisy, t, pred_noise_cur)

                visualize_dualseg(
                    output_dir=output_dir,
                    latent_noisy=latent_noisy, latent_gt=latent_gt, latent_recon=latent_recon,
                    edge_mask_pred=edge_pred, edge_mask_pred_aux=edge_pred_aux, edge_gt_mask=edge_gt,
                    parcel_mask_pred=parcel_pred, parcel_mask_pred_aux=parcel_pred_aux, parcel_gt_mask=parcel_gt,
                    images=images, iteration=iteration, t=t
                )

        extra_info = {}
        return loss_dict, extra_info

    def forward_test(self, images, annotations=None, sampler="direct", ddim_steps=200):
        inputs = self.backbone(images)
        features = self.uper_head(inputs)
        features_aux = self.auxiliary_head(inputs)

        # Existing edge branch.
        edge_feature = features
        edge_pred = self.mask_predictor(edge_feature)
        edge_pred = F.interpolate(edge_pred, scale_factor=4, mode='bilinear', align_corners=False)
        edge_mask_prob = edge_pred.softmax(dim=1)
        edge_seg_mask = edge_mask_prob.argmax(dim=1)

        # Parcel branch.
        parcel_feature = self.parcel_mask_head(features)
        parcel_pred = self.parcel_mask_predictor(parcel_feature)
        parcel_pred = F.interpolate(parcel_pred, scale_factor=4, mode='bilinear', align_corners=False)
        parcel_mask_prob = parcel_pred.softmax(dim=1)
        parcel_seg_mask = parcel_mask_prob.argmax(dim=1)

        # The LDM branch stays unchanged.
        cond = features
        bsz, _, h4, w4 = features.shape
        latent_shape = (bsz, 3, h4, w4)

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

        output = {
            "vertex_heatmap_latent_recon": latent_recon,

            # Edge branch outputs; keep original aliases for downstream compatibility.
            "mask_prob": edge_mask_prob,
            "seg_mask": edge_seg_mask,
            "edge_mask_prob": edge_mask_prob,
            "edge_seg_mask": edge_seg_mask,

            # Parcel branch outputs.
            "parcel_mask_prob": parcel_mask_prob,
            "parcel_seg_mask": parcel_seg_mask,
        }
        extra_info = {}

        return output, extra_info


def visualize_dualseg(output_dir, latent_noisy, latent_gt, latent_recon,
                      edge_mask_pred, edge_mask_pred_aux, edge_gt_mask,
                      parcel_mask_pred, parcel_mask_pred_aux, parcel_gt_mask,
                      images, iteration, t):
    # Reuse the original helper for image, latent, and edge-mask visualization.
    visualize(
        output_dir=output_dir,
        latent_noisy=latent_noisy,
        latent_gt=latent_gt,
        latent_recon=latent_recon,
        mask_pred=edge_mask_pred,
        mask_pred_aux=edge_mask_pred_aux,
        gt_mask=edge_gt_mask,
        images=images,
        iteration=iteration,
        t=t,
    )

    parcel_mask_dir = os.path.join(output_dir, "parcel_mask")
    os.makedirs(parcel_mask_dir, exist_ok=True)

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

    print(f"[Visualize] Saved parcel-mask visualization for iteration {iteration} to: {parcel_mask_dir}")


class DualSegAnnotationFormatter(object):
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
            'mask': ann['mask'],
            'parcel_mask': ann['parcel_mask'],
        }
        meta = {}
        return target, meta

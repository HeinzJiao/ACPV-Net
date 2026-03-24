#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPerNet detector with deconvolution-style vertex heatmap decoding.

This legacy variant explores alternative upsampling heads and optional
geometry-aware training utilities.
"""

import cv2
import torch
import torch.nn.functional as F
from math import log
from torch import nn
from topomapper.backbones import build_backbone
from topomapper.utils.polygon import generate_polygon
from topomapper.utils.polygon import get_pred_junctions
from skimage.measure import label, regionprops
from topomapper.uper_head import UPerHead
from topomapper.fcn_head_feat import FCNHead
from topomapper.eca import ECA
from mmcv.cnn import ConvModule
from torch.utils.data.dataloader import default_collate
from topomapper.losses import junction_loss, geometric_consistency_losses, save_heatmap_color, jloc_loss_edge_band  # Includes the edge-band-supervised junction loss.
import os
import numpy as np
import cv2


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)

    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        t = ((mask == 1) | (mask == 2)).float()
        w = t.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(t/w)

    return loss.mean()

class BuildingUPerNetDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingUPerNetDetector, self).__init__()

        # --------------------
        # Backbone
        # --------------------
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME

        self.num_classes = cfg.DATASETS.NUM_CLASSES  # Supports multi-class masks.

        if not test:
            self.formatter = AnnotationFormatter(cfg)

        self.target_height = cfg.DATASETS.TARGET.HEIGHT
        self.target_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        # ---------------------------------
        # Main semantic decoder (UPerHead)
        # ---------------------------------
        decode_cfg = cfg.MODEL.DECODE_HEAD
        self.uper_head = UPerHead(
            in_channels=decode_cfg.IN_CHANNELS,  # e.g., [96, 192, 384, 768]
            in_index=decode_cfg.IN_INDEX,  # e.g., [0, 1, 2, 3]
            pool_scales=decode_cfg.POOL_SCALES,  # e.g., [1, 2, 3, 6]
            channels=decode_cfg.CHANNELS,  # e.g., 512
            dropout_ratio=decode_cfg.DROPOUT_RATIO,  # e.g., 0.1
            num_classes=self.num_classes,  # Usually set from the outer config.
            align_corners=decode_cfg.ALIGN_CORNERS  # e.g., False
        )
        self.channels = decode_cfg.CHANNELS

        # ---------------------------
        # Auxiliary branch (FCNHead)
        # ---------------------------
        aux_cfg = cfg.MODEL.AUX_HEAD
        self.auxiliary_head = FCNHead(
            in_channels=aux_cfg.IN_CHANNELS,  # e.g., 384
            in_index=aux_cfg.IN_INDEX,  # e.g., 2
            channels=aux_cfg.CHANNELS,  # e.g., 256
            num_convs=aux_cfg.NUM_CONVS,  # e.g., 1
            concat_input=aux_cfg.CONCAT_INPUT,  # e.g., False
            dropout_ratio=aux_cfg.DROPOUT_RATIO,  # e.g., 0.1
            num_classes=aux_cfg.NUM_CLASSES,  # e.g., 7
            align_corners=aux_cfg.ALIGN_CORNERS  # e.g., False
        )

        # ----------------------------------
        # Additional processing of UPerHead features
        # ----------------------------------
        self.mask_head = self._make_conv(self.channels, self.channels, self.channels)
        self.jloc_head = self._make_conv(self.channels, self.channels, self.channels)

        # -----------------------
        # Main prediction heads
        # -----------------------
        # Keep the original mask predictor.
        self.mask_predictor = self._make_predictor(self.channels, self.num_classes)

        # Optional ViTPose-style deconvolution decoder for the junction branch.
        # self.jloc_upsample_head = self._make_deconv_predictor(
        #     dim_in=self.channels,
        #     dim_out=1,
        #     num_deconv_layers=2,
        #     num_deconv_filters=(256, 256),
        #     num_deconv_kernels=(4, 4)
        # )

        coarse_to_fine=cfg.MODEL.coarse_to_fine

        # Junction branch with a lighter ViTPose-style bilinear decoder.
        if coarse_to_fine:
            self.jloc_upsample_head = JLocHead(in_ch=self.channels, aux_supervision=False)
        else:
            self.jloc_upsample_head = nn.Sequential(
                nn.ReLU(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(self.channels, 1, kernel_size=3, padding=1)
            )

        # Auxiliary prediction heads with a lightweight ViTPose-style decoder.
        if aux_cfg.DROPOUT_RATIO > 0:
            self.dropout = nn.Dropout2d(aux_cfg.DROPOUT_RATIO)
        else:
            self.dropout = nn.Identity()

        # -----------------------
        # Auxiliary prediction modules
        # -----------------------
        self.mask_aux_upsample_head = nn.Sequential(
            self.dropout,
            nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(aux_cfg.CHANNELS, aux_cfg.NUM_CLASSES, kernel_size=3, padding=1)
        )

        self.jloc_aux_upsample_head = nn.Sequential(
            self.dropout,
            nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(aux_cfg.CHANNELS, 1, kernel_size=3, padding=1)
        )

        # # mask branch -><- vertex heatmap branch
        # self.m2j_att = ECA(self.channels)
        # self.j2m_att = ECA(self.channels)

        # Track the training iteration.
        self.train_step = 0
        
    def forward(self, images, annotations = None, iteration=None, output_dir=None):
        if self.training:
            return self.forward_train(images, annotations=annotations,
                                      iteration=iteration, output_dir=output_dir)
        else:
            return self.forward_test(images, annotations=annotations, apply_sigmoid=True)

    def forward_train(self, images, annotations = None, iteration=None, output_dir=None):
        """Forward pass during training."""
        self.train_step += 1

        # Extract backbone features.
        inputs = self.backbone(images)

        # Main branch outputs from UPerHead.
        features = self.uper_head(inputs)  # torch.Size([B, 512, 128, 128])
        mask_pred, jloc_pred = self.predict(features)  # torch.Size([B, 7, , ]), torch.Size([B, 1, , ])

        # Auxiliary branch outputs from FCNHead.
        features_aux = self.auxiliary_head(inputs)  # torch.Size([B, 256, 32, 32])
        mask_pred_aux, jloc_pred_aux = self.predict_aux(features_aux)  # torch.Size([B, 7, , ]), torch.Size([B, 1, , ])

        # Encode ground truth annotations.
        targets, metas = self.formatter(annotations)

        # Initialize the loss dictionary.
        loss_dict = {
            'loss_mask': 0.0,
            'loss_jloc': 0.0,
            'loss_mask_aux': 0.0,
            'loss_jloc_aux': 0.0,
        }

        if targets is not None:
            loss_dict = self.compute_loss(
                images,
                mask_pred, jloc_pred,
                mask_pred_aux, jloc_pred_aux,
                targets, loss_dict,
                use_geometric_consistency=False,
                use_edge_supervised_jloc=False,
                apply_edge_sup_to_aux=False,
                iteration=iteration, output_dir=output_dir,
                visualize_every=5000
            )

        extra_info = {}
        return loss_dict, extra_info

    def predict(self, features):
        """Predict the segmentation mask and junction heatmap."""
        # Feature heads
        mask_feature = self.mask_head(features)  # F_seg
        jloc_feature = self.jloc_head(features)  # F_ver

        # # bidirectional fusion
        # mask_att_feature = self.j2m_att(jloc_feature, mask_feature)
        # jloc_att_feature = self.m2j_att(mask_feature, jloc_feature)

        mask_pred = self.mask_predictor(mask_feature)
        mask_pred = F.interpolate(mask_pred, scale_factor=4, mode='bilinear', align_corners=False)
        # mask_pred = self.mask_upsample_head(mask_feature)  # Outputs full-resolution logits.

        jloc_pred = self.jloc_upsample_head(jloc_feature)  # Outputs full-resolution logits.

        return mask_pred, jloc_pred

    def predict_aux(self, features):
        """
        Auxiliary prediction with a lightweight ViTPose-style decoder:
        Conv3x3(Bilinear(ReLU(Feature))) with leading dropout.
        """
        mask_pred_aux = self.mask_aux_upsample_head(features)
        jloc_pred_aux = self.jloc_aux_upsample_head(features)
        return mask_pred_aux, jloc_pred_aux

    def compute_loss(self, images, mask_pred, jloc_pred, mask_pred_aux, jloc_pred_aux,
                     targets, loss_dict,
                     use_geometric_consistency=False,
                     use_edge_supervised_jloc=False,  # Whether to apply edge-band supervision to the main branch.
                     apply_edge_sup_to_aux=False,  # Whether to apply edge-band supervision to the auxiliary branch.
                     edge_sup_cfg=None,  # Optional configuration dictionary for edge-band supervision.
                     iteration=None, output_dir=None, visualize_every=1000):
        """
        Compute losses for the main and auxiliary branches.
        `jloc_pred` denotes junction logits before sigmoid normalization.
        """
        # -------------------
        # Segmentation losses
        # -------------------
        gt_mask_long = targets['mask'].squeeze(1).long()
        loss_dict['loss_mask'] = F.cross_entropy(mask_pred, gt_mask_long)
        loss_dict['loss_mask_aux'] = F.cross_entropy(mask_pred_aux, gt_mask_long)

        # --------------------------------
        # Junction heatmap loss (main branch)
        # --------------------------------
        if use_edge_supervised_jloc:
            # Edge-band supervision inside the mixed boundary band.
            out_main = jloc_loss_edge_band(
                jloc_pred=jloc_pred,
                mask_pred_logits=mask_pred,
                gt_mask_long=gt_mask_long,
                junc_target=targets['junc_heatmap'],
                iteration=iteration,
                **(edge_sup_cfg or {})
            )
            loss_dict['loss_jloc_mse'] = out_main['mse']  # Keep the same key for LOSS_WEIGHTS compatibility.
            for k in ('dbg_k_inband_ratio', 'dbg_band_coverage'):
                if k in out_main:
                    loss_dict[k] = out_main[k]
        else:
            # Original junction supervision (MSE / KL / related terms).
            loss_jloc = junction_loss(
                pred=jloc_pred,
                target=targets['junc_heatmap'],
                use_bce_focal=False,
                use_kl=False,
                use_focal=False,
                apply_sigmoid=True
            )
            if 'mse' in loss_jloc:       loss_dict['loss_jloc_mse'] = loss_jloc['mse']
            if 'mse_focal' in loss_jloc:       loss_dict['loss_jloc_mse_focal'] = loss_jloc['mse_focal']
            if 'bce_focal' in loss_jloc: loss_dict['loss_jloc_bce_focal'] = loss_jloc['bce_focal']
            if 'kl' in loss_jloc:        loss_dict['loss_jloc_kl'] = loss_jloc['kl']
            if 'entropy' in loss_jloc:   loss_dict['loss_jloc_entropy'] = loss_jloc['entropy']

        # -----------------------------------
        # Junction heatmap loss (auxiliary branch)
        # -----------------------------------
        loss_jloc_aux = junction_loss(
            pred=jloc_pred_aux,
            target=targets['junc_heatmap'],
            use_bce_focal=False,
            use_kl=False,
            use_focal=False,
            apply_sigmoid=True
        )
        if 'mse' in loss_jloc_aux: loss_dict['loss_jloc_aux_mse'] = loss_jloc_aux['mse']
        if 'mse_focal' in loss_jloc_aux:       loss_dict['loss_jloc_mse_focal'] = loss_jloc_aux['mse_focal']
        if 'bce_focal' in loss_jloc_aux: loss_dict['loss_jloc_aux_bce_focal'] = loss_jloc_aux['bce_focal']
        if 'kl' in loss_jloc_aux:      loss_dict['loss_jloc_aux_kl'] = loss_jloc_aux['kl']
        if 'entropy' in loss_jloc_aux: loss_dict['loss_jloc_aux_entropy'] = loss_jloc_aux['entropy']

        # ---------------------
        # Geometric consistency
        # ---------------------
        if use_geometric_consistency:
            geom_main = geometric_consistency_losses(
                mask_pred=mask_pred, jloc_pred=jloc_pred,
                lambda_edge_vertex=1.0,  # A reasonable starting point.
                lambda_dir=0.2,  # Often tuned within 0.1-0.3.
                smooth_sigma_k=1.0, smooth_sigma_e=1.0,
                iteration=iteration, output_dir=output_dir,
                visualize_every=visualize_every
            )
            for k, v in geom_main.items():
                loss_dict[k] = v

            # # Optional geometric consistency for the auxiliary branch.
            # geom_aux = geometric_consistency_losses(
            #     mask_pred=mask_pred_aux, jloc_pred=jloc_pred_aux,
            #     lambda_edge_vertex=1.0, lambda_dir=0.2,
            #     smooth_sigma_k=1.0, smooth_sigma_e=1.0
            # )
            # # Add a suffix to avoid key collisions.
            # for k, v in geom_aux.items():
            #     loss_dict[k + "_aux"] = v

        # -----------------------------------
        # Intermediate visualization (every N iterations)
        # -----------------------------------
        if iteration is not None and iteration % visualize_every == 0:
            with torch.no_grad():
                visualize(output_dir=output_dir,
                          jloc_pred=jloc_pred, jloc_pred_aux=jloc_pred_aux,
                          mask_pred=mask_pred, mask_pred_aux=mask_pred_aux, gt_mask=targets['mask'].squeeze(1).long(),
                          images=images,
                          iteration=iteration,
                          apply_sigmoid=True)

        return loss_dict

    def _make_deconv_predictor(self, dim_in, dim_out,
                               num_deconv_layers=2,
                               num_deconv_filters=(256, 256),
                               num_deconv_kernels=(4, 4),
                               final_kernel_size=1):
        """
        Build a deconvolution-based prediction head that upsamples features
        from H/4 to full resolution.
        - dim_in: input channel count from the backbone
        - dim_out: output channel count, e.g. mask classes or a single heatmap
        """
        layers = []
        in_channels = dim_in
        assert len(num_deconv_filters) == num_deconv_layers
        assert len(num_deconv_kernels) == num_deconv_layers

        for i in range(num_deconv_layers):
            out_channels = num_deconv_filters[i]
            kernel_size = num_deconv_kernels[i]
            padding = (kernel_size - 2) // 2
            output_padding = 0 if kernel_size == 4 else 1

            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        layers.append(
            nn.Conv2d(
                in_channels,
                dim_out,
                kernel_size=final_kernel_size,
                stride=1,
                padding=(final_kernel_size - 1) // 2)
        )

        return nn.Sequential(*layers)

    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
                    nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, dim_out, kernel_size=1),
                )
        return layer

    def forward_test(self, images, annotations = None, apply_sigmoid=False):
        # Extract backbone features.
        inputs = self.backbone(images)

        # Main branch outputs from UPerHead.
        features = self.uper_head(inputs)  # torch.Size([B, 512, 128, 128])
        mask_pred, jloc_pred = self.predict(features)  # (B, num_classes, H, W), (B, 2, H, W)

        mask_prob = mask_pred.softmax(dim=1)  # (B, num_classes, H, W)
        seg_mask = mask_prob.argmax(dim=1)  # (B, H, W)，int64

        if apply_sigmoid:
            jloc_pred = torch.sigmoid(jloc_pred)

        jloc_pred = jloc_pred.squeeze(1)  # (B, H, W)

        output = {
            'mask_prob': mask_prob,  # (B, num_classes, H, W), per-class probability map.
            'seg_mask': seg_mask,  # (B, H, W), int64 segmentation label map.
            'jloc_prob': jloc_pred,  # (B, H, W), float junction probability map.
        }
        extra_info = {}

        return output, extra_info


class JLocHead(nn.Module):
    def __init__(self, in_ch, mid1=None, mid2=None, aux_supervision=False):
        super().__init__()
        mid1 = mid1 if mid1 else in_ch // 2
        mid2 = mid2 if mid2 else max(64, in_ch // 4)

        # H/4 -> H/2
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, mid1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid1),
            nn.ReLU(),
        )
        # Optional auxiliary output at H/2.
        self.aux_head = nn.Conv2d(mid1, 1, kernel_size=1) if aux_supervision else None

        # H/2 -> H
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid1, mid2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(),
        )

        # Final logits without ReLU.
        self.out = nn.Conv2d(mid2, 1, kernel_size=1)

    def forward(self, x):
        x_h2 = self.up1(x)  # H/2
        aux = self.aux_head(x_h2) if self.aux_head else None

        x = self.up2(x_h2)  # H
        logits = self.out(x)  # H logits

        if self.aux_head:
            return logits, aux
        else:
            return logits


class AnnotationFormatter(object):
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
            'mask': ann['mask'][None],  # (1, 512, 512)
            'junc_heatmap': ann['junc_heatmap'][None],
        }

        meta = {
        }

        return target, meta

def visualize(output_dir, jloc_pred, jloc_pred_aux, mask_pred, mask_pred_aux, gt_mask, images, iteration, apply_sigmoid=False):
    """
    Visualize predicted heatmaps and masks together with the input image.
    Args:
        output_dir (str): Root output directory.
        images (Tensor): [B, 3, H, W] normalized input images.
        iteration (int): Current iteration index used in file names.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "jloc"), exist_ok=True)

    # Use the first item in the batch.
    image_np = images[0].detach().cpu().numpy()               # [3, H, W]

    # ----------- Visualize the input image (after de-normalization) -----------
    pixel_mean = np.array([109.730, 103.832, 98.681]).reshape(3, 1, 1)
    pixel_std  = np.array([22.275, 22.124, 23.229]).reshape(3, 1, 1)

    image_denorm = image_np * pixel_std + pixel_mean           # De-normalize.
    image_denorm = np.clip(image_denorm, 0, 255).astype(np.uint8)
    image_vis = np.transpose(image_denorm, (1, 2, 0))          # CHW -> HWC for OpenCV.
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)     # Convert to BGR before saving.

    save_path_img = os.path.join(output_dir, "image", f'image_{iteration}.png')
    cv2.imwrite(save_path_img, image_vis)

    # === Parse mask_pred, mask_pred_aux, and gt_mask ===
    mask_pred_prob = F.softmax(mask_pred, dim=1)
    seg_mask_pred = mask_pred_prob.argmax(dim=1)[0].detach().cpu().numpy()  # [H, W]

    mask_pred_aux_prob = F.softmax(mask_pred_aux, dim=1)
    seg_mask_pred_aux = mask_pred_aux_prob.argmax(dim=1)[0].detach().cpu().numpy()  # [H, W]

    seg_mask_gt = gt_mask[0].detach().cpu().numpy()  # (H, W), already squeezed.

    # === Visualization: class ID -> RGB mapping ===
    id_to_color = {
        0: (255, 0, 0),  # artificial_structure
        1: (165, 42, 42),  # building
        2: (255, 255, 0),  # road
        3: (128, 128, 128),  # unvegetated
        4: (0, 255, 0),  # vegetation
        5: (0, 0, 255),  # water
        6: (0, 0, 0)  # unknown
    }

    def mask_to_rgb(mask):
        H, W = mask.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for id, color in id_to_color.items():
            rgb[mask == id] = color
        return rgb

    # Save the three mask visualizations.
    cv2.imwrite(os.path.join(output_dir, "mask", f"mask_pred_{iteration}.png"), mask_to_rgb(seg_mask_pred))
    cv2.imwrite(os.path.join(output_dir, "mask", f"mask_pred_aux_{iteration}.png"), mask_to_rgb(seg_mask_pred_aux))
    cv2.imwrite(os.path.join(output_dir, "mask", f"mask_gt_{iteration}.png"), mask_to_rgb(seg_mask_gt))

    # === Visualize jloc_pred (junction heatmap) ===
    if apply_sigmoid:
        jloc_pred = torch.sigmoid(jloc_pred)
        jloc_pred_aux = torch.sigmoid(jloc_pred_aux)

    save_heatmap_color(jloc_pred[0],  # (1,H,W)
                       os.path.join(output_dir, "jloc", f"jloc_pred_{iteration}.png"),
                       cmap="jet", norm="minmax", value_range=(0.0, 1.0))

    save_heatmap_color(jloc_pred_aux[0],  # (1,H,W)
                       os.path.join(output_dir, "jloc", f"jloc_pred_aux_{iteration}.png"),
                       cmap="jet", norm="minmax", value_range=(0.0, 1.0))

    # === Logging ===
    print(f"[Visualize] mask visualization for iteration {iteration} to: {output_dir}")

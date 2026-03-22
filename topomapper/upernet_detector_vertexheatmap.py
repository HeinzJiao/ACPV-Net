#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPerNet detector with direct vertex heatmap prediction.

This legacy variant predicts semantic masks and vertex heatmaps without LDM.
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
from topomapper.fcn_head import FCNHead
from topomapper.uper_head import PPM
from topomapper.eca import ECA
from mmcv.cnn import ConvModule
from topomapper.csrc.lib.afm_op import afm
from torch.utils.data.dataloader import default_collate


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
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME

        self.num_classes = cfg.DATASETS.NUM_CLASSES  # Supports multi-class masks.

        if not test:
            self.formatter = AnnotationFormatter(cfg)

        self.target_height = cfg.DATASETS.TARGET.HEIGHT
        self.target_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        # UPerHead configuration.
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

        # Auxiliary FCNHead configuration.
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

        self.channels = decode_cfg.CHANNELS

        # Feature heads
        self.mask_head = self._make_conv(self.channels, self.channels, self.channels)
        self.jloc_head = self._make_conv(self.channels, self.channels, self.channels)

        # Seg Head
        self.mask_predictor = self._make_predictor(self.channels, self.num_classes)
        # Ver Head
        self.jloc_predictor = self._make_predictor(self.channels, 1)

        self.train_step = 0
        
    def forward(self, images, annotations = None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations)

    def forward_train(self, images, annotations = None):
        """Forward pass during training."""
        self.train_step += 1

        # Extract backbone features.
        inputs = self.backbone(images)

        # Main branch outputs from UPerHead.
        features = self.uper_head(inputs)  # torch.Size([B, 512, 128, 128])
        mask_pred, jloc_pred = self.predict(features)  # torch.Size([B, 7, 128, 128]), torch.Size([B, 1, 128, 128])

        # Auxiliary branch outputs from FCNHead.
        mask_pred_aux, jloc_pred_aux = self.auxiliary_head(inputs)  # torch.Size([B, 7, 32, 32]), torch.Size([B, 1, 32, 32])

        # Encode ground truth annotations.
        targets, metas = self.formatter(annotations)

        # Initialize the loss dictionary.
        loss_dict = {
            'loss_mask': 0.0,
            'loss_jloc': 0.0,
            'loss_mask_aux': 0.0,
            'loss_jloc_aux': 0.0
        }

        if targets is not None:
            loss_dict = self.compute_loss(
                mask_pred, jloc_pred,
                mask_pred_aux, jloc_pred_aux,
                targets, loss_dict
            )

        extra_info = {}
        return loss_dict, extra_info

    def predict(self, features):
        """Predict the segmentation mask and junction heatmap."""
        # Feature heads
        mask_feature = self.mask_head(features)  # F_seg
        jloc_feature = self.jloc_head(features)  # F_ver

        # Seg Head
        mask_pred = self.mask_predictor(mask_feature)  # shape: (B, num_classes, H, W)

        # Ver Head
        jloc_pred = self.jloc_predictor(jloc_feature)  # (B, 1, H, W)

        return mask_pred, jloc_pred

    def compute_loss(self, mask_pred, jloc_pred, mask_pred_aux, jloc_pred_aux, targets, loss_dict):
        """Compute losses for the main and auxiliary branches."""
        # mask_pred:  torch.Size([2, 7, 128, 128])
        # jloc_pred:  torch.Size([2, 1, 128, 128])
        # mask_pred_aux:  torch.Size([2, 7, 32, 32])
        # jloc_pred_aux:  torch.Size([2, 1, 32, 32])

        mask_pred = F.interpolate(mask_pred, size=(self.origin_height, self.origin_width), mode='bilinear',
                                  align_corners=False)
        jloc_pred = F.interpolate(jloc_pred, size=(self.origin_height, self.origin_width), mode='bilinear',
                                  align_corners=False)

        loss_dict['loss_mask'] = F.cross_entropy(mask_pred, targets['mask'].squeeze(1).long())
        loss_dict['loss_jloc'] = F.l1_loss(jloc_pred, targets['junc_heatmap'])

        mask_pred_aux = F.interpolate(mask_pred_aux, size=(self.origin_height, self.origin_width), mode='bilinear',
                                  align_corners=False)
        jloc_pred_aux = F.interpolate(jloc_pred_aux, size=(self.origin_height, self.origin_width), mode='bilinear',
                                  align_corners=False)

        loss_dict['loss_mask_aux'] = F.cross_entropy(mask_pred_aux, targets['mask'].squeeze(1).long())
        loss_dict['loss_jloc_aux'] = F.l1_loss(jloc_pred_aux, targets['junc_heatmap'])

        return loss_dict
    
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

    def forward_test(self, images, annotations = None):
        # Extract backbone features.
        inputs = self.backbone(images)

        # Main branch outputs from UPerHead.
        features = self.uper_head(inputs)  # torch.Size([B, 512, 128, 128])
        mask_pred, jloc_pred = self.predict(features)  # (B, num_classes, H, W), (B, 2, H, W)

        # Upsample segmentation logits before softmax for smoother contours.
        mask_pred_up = F.interpolate(mask_pred, size=(self.origin_height, self.origin_width),
                                     mode='bilinear', align_corners=False)
        mask_prob = mask_pred_up.softmax(dim=1)  # (B, num_classes, H, W)
        seg_mask = mask_prob.argmax(dim=1)  # (B, H, W), int64

        # # Extract all junctions.
        # junctions_all = []  # Stores per-image junction coordinates.
        # scale_y = self.origin_height / self.target_height
        # scale_x = self.origin_width / self.target_width
        #
        # for b in range(jloc_prob.size(0)):
        #     juncs_pred_single = self.get_pred_junctions(jloc_prob[b])
        #     juncs_pred_single[:, 0] *= scale_x
        #     juncs_pred_single[:, 1] *= scale_y
        #     junctions_all.append(juncs_pred_single)

        jloc_pred_up = F.interpolate(jloc_pred, size=(self.origin_height, self.origin_width),
                                     mode='bilinear', align_corners=False)
        jloc_pred_up = jloc_pred_up.squeeze(1)  # (B, H, W)
        """"""

        output = {
            'mask_prob': mask_prob,  # (B, num_classes, H, W), per-class probability map.
            'seg_mask': seg_mask,  # (B, H, W), int64 segmentation label map.
            'jloc_prob': jloc_pred_up,  # (B, H, W), float junction heatmap.
            # 'junctions': junctions_all  # list of np.ndarray with shape (N_i, 2).
        }
        extra_info = {}

        return output, extra_info

    def non_maximum_suppression(self, a):
        ap = F.max_pool2d(a, 3, stride=1, padding=1)
        mask = (a == ap).float().clamp(min=0.0)
        return a * mask

    def get_junctions(self, jloc, topk=300, th=0):
        height, width = jloc.shape
        jloc = jloc.reshape(-1)

        scores, index = torch.topk(jloc, k=topk)
        y = index // width
        x = index % width

        junctions = torch.stack((x, y)).t()

        return junctions[scores > th], scores[scores > th]

    def get_pred_junctions(self, jloc_prob_single, score_thresh=0.008, topk=300):
        # Apply non-maximum suppression first.
        jloc_nms = self.non_maximum_suppression(jloc_prob_single)  # (1, H, W); only local peaks remain.
        jloc_nms = jloc_nms.squeeze(0)

        # Count candidates above the score threshold and cap them by top-k.
        num_candidate = int((jloc_nms > score_thresh).float().sum().item())
        topk = min(topk, num_candidate)

        # Retrieve the coordinates of the top-k junction candidates.
        junctions, _ = self.get_junctions(jloc_nms, topk=topk)  # (N, 2)

        return junctions.detach().cpu().numpy()  # Return NumPy coordinates.


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

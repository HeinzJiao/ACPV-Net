#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight FCN head used as a feature refinement block.

This implementation returns intermediate features rather than final class
logits, and is used as an auxiliary head in several detector variants.
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F


class FCNHead(nn.Module):
    def __init__(self,
                 in_channels=384,
                 in_index=2,
                 channels=256,
                 num_convs=1,
                 kernel_size=3,
                 concat_input=False,
                 dilation=1,
                 dropout_ratio=0.1,
                 num_classes=7,
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)

        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        self.concat_input = concat_input
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        if num_convs == 0:
            assert self.in_channels == self.channels
            self.convs = nn.Identity()
        else:
            convs.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            for i in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)


    def _transform_inputs(self, inputs):
        return inputs[self.in_index]

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        features = self._forward_feature(inputs)
        return features

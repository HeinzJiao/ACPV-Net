#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform utilities for image-annotation preprocessing.

This module provides resize, tensor conversion, normalization, and simple
color jitter operations used by the dataset builders.
"""

import cv2
import torch
import torchvision
import numpy as np
from torchvision.transforms import functional as F
from skimage.transform import resize

class Resize(object):
    """
    Training-time preprocessing that jointly resizes the image and annotations.

    Used by:
        - Mask-only baseline variants
          -> `topomapper/upernet_detector.py`
        - Direct vertex-heatmap variants without LDM
          -> `topomapper/upernet_detector_vertexheatmap.py`
          -> `topomapper/upernet_detector_vertexheatmap_deconv.py`
        - Vertex-heatmap + AFM variants without LDM
          -> `topomapper/upernet_detector_vertexheatmap_afm.py`

    Args:
        image_height (int): Target image height.
        image_width (int): Target image width.
        ann_height (int): Target annotation height.
        ann_width (int): Target annotation width.

    Behavior:
        - Resize the image if the size differs from the target size.
        - Normalize pixel values to `[0, 1]`.
        - Scale and clip junction coordinates, while preserving the original
          coordinates as `junc_ori`.
        - Resize the mask and preserve the original mask as `mask_ori`.
        - Update annotation width and height.

    Returns:
        The normalized image and the updated annotation dictionary.
    """
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width = image_width
        self.ann_height = ann_height
        self.ann_width = ann_width

    def __call__(self, image, ann):
        h, w = image.shape[:2]

        # Skip resizing if the size already matches the target shape.
        if (h, w) != (self.image_height, self.image_width):
            image = resize(image, (self.image_height, self.image_width))

        image = np.array(image, dtype=np.float32) / 255.0

        sx = self.ann_width / ann['width']
        sy = self.ann_height / ann['height']
        ann['junc_ori'] = ann['junctions'].copy()
        ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0, self.ann_width - 1e-4)
        ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0, self.ann_height - 1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height
        ann['mask_ori'] = ann['mask'].copy()

        if ann['mask'].shape[:2] != (self.ann_height, self.ann_width):
            ann['mask'] = cv2.resize(ann['mask'].astype(np.uint8),
                                     (int(self.ann_width), int(self.ann_height)),
                                     interpolation=cv2.INTER_NEAREST)

        # Set the auxiliary resolution to 1/16 of the original annotation size.
        ann_height_aux = ann['mask_ori'].shape[0] // 16
        ann_width_aux = ann['mask_ori'].shape[1] // 16

        # Build `junc_aux` by downscaling the original junction coordinates to 1/16 resolution.
        ann['junc_aux'] = ann['junc_ori'].copy()
        ann['junc_aux'][:, 0] = np.clip(ann['junc_aux'][:, 0] / 16.0, 0, ann_width_aux - 1e-4)
        ann['junc_aux'][:, 1] = np.clip(ann['junc_aux'][:, 1] / 16.0, 0, ann_height_aux - 1e-4)

        # Resize mask_aux（1/16）
        ann['mask_aux'] = cv2.resize(ann['mask_ori'].astype(np.uint8),
                                     (ann_width_aux, ann_height_aux),
                                     interpolation=cv2.INTER_NEAREST)

        # ann:
        # mask_ori:  (512, 512)
        # mask:  (128, 128)
        # junc_ori max min:  512 0
        # junctions max min:  127 0
        # width, height: 128 128

        return image, ann


class ResizeImage(object):
    """
    Test-time preprocessing that resizes and normalizes only the image.

    Used by:
        - All test-time variants in the current pipeline
          -> `scripts/test.py`
        - LDM-based training variants that keep annotation tensors at their
          original resolution
          -> `topomapper/upernet_detector_vh_ldm.py`
          -> `topomapper/upernet_detector_vh_m_ldm.py`
          -> `topomapper/upernet_detector_vh_m_ldm_dualseg.py`
          -> `topomapper/upernet_detector_vh_m_ldm_dualseg_jointlatent.py`

    Args:
        image_height (int): Target image height.
        image_width (int): Target image width.

    Behavior:
        - Resize the image if the size differs from the target size.
        - Normalize pixel values to `[0, 1]`.
        - Return the annotation unchanged if it is provided.

    Returns:
        The normalized image, or the image-annotation pair.
    """
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image, ann=None):
        h, w = image.shape[:2]

        # Skip resizing if the size already matches the target shape.
        if (h, w) != (self.image_height, self.image_width):
            image = resize(image, (self.image_height, self.image_width))

        image = np.array(image, dtype=np.float32) / 255.0
        if ann is None:
            return image
        return image, ann


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ann=None):
        if ann is None:
            for t in self.transforms:
                image = t(image)
            return image
        for t in self.transforms:
            image, ann = t(image, ann)
        return image, ann


class ToTensor(object):
    def __call__(self, image, anns=None):
        if anns is None:
            return F.to_tensor(image)

        for key, val in anns.items():
            if isinstance(val, np.ndarray):
                anns[key] = torch.from_numpy(val.copy())
        return F.to_tensor(image), anns


class Normalize(object):
    def __init__(self, mean, std, to_255=False):
        self.mean = mean
        self.std = std
        self.to_255 = to_255

    def __call__(self, image, anns=None):
        if self.to_255:
            image *= 255.0
        image = F.normalize(image, mean=self.mean, std=self.std)
        if anns is None:
            return image
        return image, anns

class Color_jitter(object):
    def __init__(self):
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=.5, hue=.1)
    def __call__(self, image, anns=None):
        image = self.jitter(image)
        if anns is None:
            return image
        return image, anns

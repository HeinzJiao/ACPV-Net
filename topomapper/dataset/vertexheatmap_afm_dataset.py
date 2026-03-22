#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datasets for vertex heatmap supervision with AFM targets.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from .transforms import Compose, Resize, ToTensor, Normalize


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


class VertexHeatmapAfmTrainDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, augmentations=None):
        """
        Training dataset that loads images, masks, vertex heatmaps, and AFM fields.

        Args:
            data_dir (str): Dataset root directory, e.g. `./data`.
            split (str): One of `train`, `val`, or `test`.
            transform (callable): Preprocessing transform.
            augmentations (list[str]): Enabled augmentations, e.g. `hflip`, `vflip`, or `rrotate`.
        """
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")
        self.junc_heatmap_dir = os.path.join(self.root, "junction_heatmaps_sigma-3")
        self.afm_dir = os.path.join(self.root, "afm")

        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.augmentations = augmentations if augmentations is not None else []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        base_name = os.path.splitext(img_name)[0]

        # === Load the image and annotations ===
        image = io.imread(os.path.join(self.image_dir, img_name)).astype(np.float32)[:, :, :3]
        height, width = image.shape[:2]

        mask = np.array(Image.open(os.path.join(self.mask_dir, base_name + '.png')))
        junc_heatmap = np.load(os.path.join(self.junc_heatmap_dir, base_name + '.npy')).astype(np.float32)
        afm = np.load(os.path.join(self.afm_dir, base_name + '.npy')).astype(np.float32)  # (2, H, W)

        # === Data augmentation ===
        if 'hflip' in self.augmentations and np.random.rand() < 0.5:
            image = image[:, ::-1, :]
            mask = np.fliplr(mask)
            junc_heatmap = np.fliplr(junc_heatmap)

            afm = afm.transpose(1, 2, 0)  # (H, W, 2)
            afm = np.fliplr(afm)  # Horizontal flip.
            afm[..., 0] *= -1  # Flip the sign of the x component.
            afm = afm.transpose(2, 0, 1)  # Back to (2, H, W).

        if 'vflip' in self.augmentations and np.random.rand() < 0.5:
            image = image[::-1, :, :]
            mask = np.flipud(mask)
            junc_heatmap = np.flipud(junc_heatmap)

            afm = afm.transpose(1, 2, 0)  # (H, W, 2)
            afm = np.flipud(afm)  # Vertical flip.
            afm[..., 1] *= -1  # Flip the sign of the y component.
            afm = afm.transpose(2, 0, 1)  # Back to (2, H, W).

        if 'rrotate' in self.augmentations:
            angle = np.random.choice([0, 90, 180, 270])
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                junc_heatmap = cv2.rotate(junc_heatmap, cv2.ROTATE_90_CLOCKWISE)

                afm = afm.transpose(1, 2, 0)
                afm = cv2.rotate(afm, cv2.ROTATE_90_CLOCKWISE)
                afm = afm[..., [1, 0]]  # Swap dx and dy.
                afm[..., 0] *= -1  # Flip the sign of the new dx component.
                afm = afm.transpose(2, 0, 1)

            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.ROTATE_180)
                junc_heatmap = cv2.rotate(junc_heatmap, cv2.ROTATE_180)

                afm = afm.transpose(1, 2, 0)
                afm = cv2.rotate(afm, cv2.ROTATE_180)
                afm *= -1
                afm = afm.transpose(2, 0, 1)

            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                junc_heatmap = cv2.rotate(junc_heatmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

                afm = afm.transpose(1, 2, 0)
                afm = cv2.rotate(afm, cv2.ROTATE_90_COUNTERCLOCKWISE)
                afm = afm[..., [1, 0]]  # Swap dx and dy.
                afm[..., 1] *= -1  # Flip the sign of the new dy component.
                afm = afm.transpose(2, 0, 1)

        ann = {
            'mask': mask,
            'junc_heatmap': junc_heatmap,
            'afm': afm,
            'width': width,
            'height': height,
        }

        if self.transform is not None:
            return self.transform(image, ann)

        return image, ann


def collate_fn(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch]
    )

class VertexHeatmapAfmTestDataset(Dataset):
    def __init__(self, data_dir, split='test', transform=None):
        """
        Test dataset for loading images, masks, vertex heatmaps, and AFM fields.

        Args:
            data_dir (str): Dataset root directory, e.g. `./data`.
            split (str): One of `test`, `val`, or `train`.
            transform (callable): Test-time preprocessing transform.
        """
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")
        self.junc_heatmap_dir = os.path.join(self.root, "junction_heatmaps_sigma-3")
        self.afm_dir = os.path.join(self.root, "afm")

        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        base_name = os.path.splitext(img_name)[0]

        image = io.imread(os.path.join(self.image_dir, img_name)).astype(np.float32)[:, :, :3]
        height, width = image.shape[:2]

        ann = {'filename': img_name, 'height': height, 'width': width,
               'mask': np.array(Image.open(os.path.join(self.mask_dir, base_name + '.png'))),
               'junc_heatmap': np.load(os.path.join(self.junc_heatmap_dir, base_name + '.npy')).astype(np.float32),
               'afm': np.load(os.path.join(self.afm_dir, base_name + '.npy')).astype(np.float32)  # (2, H, W)
               }

        if self.transform is not None:
            return self.transform(image, ann)

        return image, ann

    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])

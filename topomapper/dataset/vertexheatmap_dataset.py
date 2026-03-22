#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datasets for direct vertex heatmap supervision.
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


class VertexHeatmapTrainDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, augmentations=None):
        """
        Training dataset that loads images, masks, and vertex heatmaps.

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

        ann = {
            'mask': mask,
            'junc_heatmap': junc_heatmap,
            'width': width,
            'height': height,
        }

        # === Data augmentation ===
        if 'hflip' in self.augmentations and np.random.rand() < 0.5:
            image = image[:, ::-1, :]
            ann['mask'] = np.fliplr(ann['mask'])
            ann['junc_heatmap'] = np.fliplr(ann['junc_heatmap'])

        if 'vflip' in self.augmentations and np.random.rand() < 0.5:
            image = image[::-1, :, :]
            ann['mask'] = np.flipud(ann['mask'])
            ann['junc_heatmap'] = np.flipud(ann['junc_heatmap'])

        if 'rrotate' in self.augmentations:
            angle = np.random.choice([0, 90, 180, 270])
            if angle == 90 or angle == 270:
                rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                image = cv2.warpAffine(image, rot_matrix, (width, height))
                ann['mask'] = cv2.warpAffine(ann['mask'], rot_matrix, (width, height))
                ann['junc_heatmap'] = cv2.warpAffine(ann['junc_heatmap'], rot_matrix, (width, height))
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
                ann['mask'] = cv2.rotate(ann['mask'], cv2.ROTATE_180)
                ann['junc_heatmap'] = cv2.rotate(ann['junc_heatmap'], cv2.ROTATE_180)

        if self.transform is not None:
            return self.transform(image, ann)

        return image, ann


def collate_fn(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch]
    )

class VertexHeatmapTestDataset(Dataset):
    def __init__(self, data_dir, split='test', transform=None):
        """
        Test dataset for loading images, masks, and vertex heatmaps.

        Args:
            data_dir (str): Dataset root directory, e.g. `./data`.
            split (str): One of `test`, `val`, or `train`.
            transform (callable): Test-time preprocessing transform.
        """
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")
        self.junc_heatmap_dir = os.path.join(self.root, "junction_heatmaps_sigma-3")

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
               'junc_heatmap': np.load(os.path.join(self.junc_heatmap_dir, base_name + '.npy')).astype(np.float32)}

        if self.transform is not None:
            return self.transform(image, ann)

        return image, ann

    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])

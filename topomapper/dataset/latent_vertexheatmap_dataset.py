#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datasets for latent vertex heatmap training and testing.

These dataset classes load images, semantic masks, and latent vertex heatmaps
for the single-branch latent heatmap setup.
"""

import os
import torch
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import cv2


class LatentVertexHeatmapTrainDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, augmentations=None):
        """
        Training dataset for images, masks, and latent vertex heatmaps.
        """
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")
        self.latent_dir = os.path.join(self.root, "heatmap_augmented_latent_kl-4")

        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.augmentations = augmentations if augmentations is not None else []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        base_name = os.path.splitext(img_name)[0]

        # Load the image and mask.
        image = io.imread(os.path.join(self.image_dir, img_name)).astype(np.float32)[:, :, :3]
        height, width = image.shape[:2]

        mask = np.array(Image.open(os.path.join(self.mask_dir, base_name + '.png')))

        # Randomly choose one augmentation aligned with the latent directory.
        valid_transforms = self.augmentations if len(self.augmentations) > 0 else ['rot0']
        latent_subdir = np.random.choice(valid_transforms)

        # Apply the selected transform to the image and mask.
        if latent_subdir == 'rot0':
            pass  # No transformation.
        elif latent_subdir == 'rot90':
            image = np.rot90(image, k=1)  # 90 degrees counter-clockwise.
            mask = np.rot90(mask, k=1)
        elif latent_subdir == 'rot180':
            image = np.rot90(image, k=2)
            mask = np.rot90(mask, k=2)
        elif latent_subdir == 'rot270':
            image = np.rot90(image, k=3)
            mask = np.rot90(mask, k=3)
        elif latent_subdir == 'flip_h':
            image = image[:, ::-1, :]
            mask = np.fliplr(mask)
        elif latent_subdir == 'flip_v':
            image = image[::-1, :, :]
            mask = np.flipud(mask)
        elif latent_subdir == 'flip_diag':
            # Main-diagonal flip: (x, y) -> (y, x).
            image = np.transpose(image, (1, 0, 2))
            mask = mask.T
        elif latent_subdir == 'flip_anti_diag':
            # Anti-diagonal flip: (x, y) -> (W-1-y, H-1-x).
            # Equivalent to transpose followed by horizontal and vertical flips.
            image = np.transpose(image, (1, 0, 2))
            mask = mask.T
            image = image[::-1, ::-1, :]
            mask = mask[::-1, ::-1]
        else:
            raise ValueError(f"Unsupported transform type: {latent_subdir}")

        # Load the latent heatmap.
        latent_path = os.path.join(self.latent_dir, latent_subdir, 'z', base_name + ".pt")
        vertex_heatmap_latent = torch.load(latent_path).float()

        ann = {
            'filename': img_name,
            'height': height,
            'width': width,
            'vertex_heatmap_latent': vertex_heatmap_latent,
            'mask': mask
        }

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

def collate_fn(batch):
    return default_collate([b[0] for b in batch]), [b[1] for b in batch]


class LatentVertexHeatmapTestDataset(Dataset):
    """Test dataset that loads images only for latent vertex heatmap inference."""

    def __init__(self, data_dir, split='test', transform=None):
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        base_name = os.path.splitext(img_name)[0]
        image = io.imread(os.path.join(self.image_dir, img_name)).astype(np.float32)[:, :, :3]
        height, width = image.shape[:2]

        ann = {
            'filename': img_name,
            'height': height,
            'width': width,
        }

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

    @staticmethod
    def collate_fn(batch):
        return default_collate([b[0] for b in batch]), [b[1] for b in batch]



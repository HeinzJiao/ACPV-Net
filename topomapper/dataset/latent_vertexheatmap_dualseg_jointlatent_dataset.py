#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datasets for dual-seg joint-latent training and testing.

These dataset classes load:
- images
- edge masks
- parcel masks
- latent vertex heatmaps
- latent edge masks

for the joint-latent diffusion setup.
"""

import os
import torch
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class LatentVertexHeatmapDualSegJointLatentTrainDataset(Dataset):
    """Training dataset for the dual-seg joint-latent setup."""

    def __init__(self, data_dir, split='train', transform=None, augmentations=None,
                 parcel_mask_dirname='parcel_masks',
                 edge_latent_dirname='edge_mask_augmented_latent_kl-4'):
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")
        self.parcel_mask_dir = os.path.join(self.root, parcel_mask_dirname)
        self.vertex_latent_dir = os.path.join(self.root, "heatmap_augmented_latent_kl-4")
        self.edge_latent_dir = os.path.join(self.root, edge_latent_dirname)

        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.augmentations = augmentations if augmentations is not None else []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        base_name = os.path.splitext(img_name)[0]

        image = io.imread(os.path.join(self.image_dir, img_name)).astype(np.float32)[:, :, :3]
        height, width = image.shape[:2]

        mask = np.array(Image.open(os.path.join(self.mask_dir, base_name + '.png')))
        parcel_mask = np.array(Image.open(os.path.join(self.parcel_mask_dir, base_name + '.png')))

        valid_transforms = self.augmentations if len(self.augmentations) > 0 else ['rot0']
        latent_subdir = np.random.choice(valid_transforms)

        if latent_subdir == 'rot0':
            pass
        elif latent_subdir == 'rot90':
            image = np.rot90(image, k=1)
            mask = np.rot90(mask, k=1)
            parcel_mask = np.rot90(parcel_mask, k=1)
        elif latent_subdir == 'rot180':
            image = np.rot90(image, k=2)
            mask = np.rot90(mask, k=2)
            parcel_mask = np.rot90(parcel_mask, k=2)
        elif latent_subdir == 'rot270':
            image = np.rot90(image, k=3)
            mask = np.rot90(mask, k=3)
            parcel_mask = np.rot90(parcel_mask, k=3)
        elif latent_subdir == 'flip_h':
            image = image[:, ::-1, :]
            mask = np.fliplr(mask)
            parcel_mask = np.fliplr(parcel_mask)
        elif latent_subdir == 'flip_v':
            image = image[::-1, :, :]
            mask = np.flipud(mask)
            parcel_mask = np.flipud(parcel_mask)
        elif latent_subdir == 'flip_diag':
            image = np.transpose(image, (1, 0, 2))
            mask = mask.T
            parcel_mask = parcel_mask.T
        elif latent_subdir == 'flip_anti_diag':
            image = np.transpose(image, (1, 0, 2))
            mask = mask.T
            parcel_mask = parcel_mask.T
            image = image[::-1, ::-1, :]
            mask = mask[::-1, ::-1]
            parcel_mask = parcel_mask[::-1, ::-1]
        else:
            raise ValueError(f"Unsupported transform type: {latent_subdir}")

        vertex_latent_path = os.path.join(self.vertex_latent_dir, latent_subdir, 'z', base_name + ".pt")
        edge_latent_path = os.path.join(self.edge_latent_dir, latent_subdir, 'z', base_name + ".pt")
        vertex_heatmap_latent = torch.load(vertex_latent_path).float()
        edge_mask_latent = torch.load(edge_latent_path).float()

        ann = {
            'filename': img_name,
            'height': height,
            'width': width,
            'vertex_heatmap_latent': vertex_heatmap_latent,
            'edge_mask_latent': edge_mask_latent,
            'mask': mask,
            'parcel_mask': parcel_mask,
        }

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann


def collate_fn(batch):
    return default_collate([b[0] for b in batch]), [b[1] for b in batch]


class LatentVertexHeatmapDualSegJointLatentTestDataset(Dataset):
    """Test dataset for the dual-seg joint-latent setup."""

    def __init__(self, data_dir, split='test', transform=None):
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
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

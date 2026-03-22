#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline training dataset for mask and junction annotation loading.
"""

import cv2
import os
import json
import random
import os.path as osp
import numpy as np
from PIL import Image
from skimage import io
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class TrainDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, augmentations=None):
        """
        Args:
            data_dir (str): Dataset root directory, e.g. `./data`.
            split (str): One of `train`, `val`, or `test`.
            transform (callable, optional): Transform applied to the image-annotation pair.
            augmentations (list[str], optional): Enabled augmentations, such as `hflip`, `vflip`, or `rrotate`.
        """
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.junc_dir = os.path.join(self.root, "junctions")
        self.mask_dir = os.path.join(self.root, "masks")

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

        with open(os.path.join(self.junc_dir, base_name + '.json'), 'r') as f:
            junctions = np.array(json.load(f), dtype=np.int32)

        seg_mask = np.array(Image.open(os.path.join(self.mask_dir, base_name + '.png')))

        ann = {
            'junctions': junctions,  # ndarray of shape (N, 2), storing all vertex coordinates as [x, y].
            'mask': seg_mask,  # ndarray of shape (H, W), semantic mask with per-pixel class indices.
            'width': width,  # Image width.
            'height': height,  # Image height.
        }

        # === Data augmentation ===
        if len(junctions) > 0:
            if 'hflip' in self.augmentations and np.random.rand() < 0.5:
                image = image[:, ::-1, :]
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['mask'] = np.fliplr(ann['mask'])

            if 'vflip' in self.augmentations and np.random.rand() < 0.5:
                image = image[::-1, :, :]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['mask'] = np.flipud(ann['mask'])

            if 'rrotate' in self.augmentations:
                angle = np.random.choice([0, 90, 180, 270])
                if angle == 90 or angle == 270:
                    rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                    image = cv2.warpAffine(image, rot_matrix, (width, height))
                    ann['mask'] = cv2.warpAffine(ann['mask'], rot_matrix, (width, height))
                    ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
                elif angle == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                    ann['mask'] = cv2.rotate(ann['mask'], cv2.ROTATE_180)
                    ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                    ann['junctions'][:, 1] = height - ann['junctions'][:, 1]

        else:
            ann['junctions'] = np.array([[0, 0]], dtype=np.float32)
            ann['mask'] = np.zeros((height, width), dtype=np.uint8)

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])

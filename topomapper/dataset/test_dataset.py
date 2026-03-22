#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline test datasets for semantic masks and polygon annotations.
"""

import cv2
import numpy as np
import os.path as osp
import torchvision.datasets as dset
import os
from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from skimage import io
import json

class TestDataset(Dataset):
    def __init__(self, data_dir, split='test', transform=None):
        """
        Args:
            data_dir (str): Dataset root directory, e.g. `./data`.
            split (str): `test`, `val`, or any custom split name.
            transform (callable, optional): Transform applied to the image-annotation pair.
        """
        self.root = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.root, "images")
        self.junc_dir = os.path.join(self.root, "junctions")
        self.mask_dir = os.path.join(self.root, "masks")

        self.image_list = sorted(os.listdir(self.image_dir))
        self.transform = transform

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
            'junctions': junctions if len(junctions) > 0 else np.array([[0, 0]], dtype=np.float32),
            'mask': seg_mask if seg_mask is not None else np.zeros((height, width), dtype=np.uint8),
            'width': width,
            'height': height,
            'filename': img_name
        }

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])

class TestDatasetWithAnnotations(dset.coco.CocoDetection):
    def __init__(self, root, ann_file, transform = None):
        super(TestDatasetWithAnnotations, self).__init__(root, ann_file)

        self.root = root
        self.ids = sorted(self.ids)
        self.id_to_img_map = {k:v for k, v in enumerate(self.ids)}
        self._transforms = transform
    
    def __getitem__(self, idx):
        img, annos = super(TestDatasetWithAnnotations, self).__getitem__(idx)
        image = np.array(img).astype(float)
        img_info = self.get_img_info(idx)
        width, height = img_info['width'], img_info['height']

        ann = {
            'filename': img_info['file_name'],
            'width': width,
            'height': height,
            'junctions': [],
            'juncs_tag': [],
            'juncs_index': [],
            'segmentations': [],
            'bbox': [],
        }

        pid = 0
        instance_id = 0
        seg_mask = np.zeros([width, height])
        for ann_per_ins in annos:
            juncs, tags = [], []
            segmentations = ann_per_ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2) # the shape of the segm is (N,2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)
                points = segm[:-1]
                junc_tags = np.ones(points.shape[0])
                if i == 0:  # outline
                    poly = Polygon(points)
                    if poly.area > 0:
                        convex_point = np.array(poly.convex_hull.exterior.coords)
                        convex_index = [(p == convex_point).all(1).any() for p in points]
                        juncs.extend(points.tolist())
                        junc_tags[convex_index] = 2    # convex point label
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask += self.coco.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask, [np.int0(interior_contour)], -1, color=0, thickness=-1)

            ann['junctions'].extend(juncs)
            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['juncs_tag'].extend(tags)
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)

        ann['mask'] = np.clip(seg_mask, 0, 1)
        
        for key,_type in (['junctions',np.float32],
                          ['juncs_tag',np.long],
                          ['juncs_index', np.long],
                          ['bbox', np.float32]):
            ann[key] = np.array(ann[key],dtype=_type)


        if self._transforms is not None:
            return self._transforms(image, ann)
        return image, ann

    def image(self, idx):
        img_id = self.id_to_img_map[idx]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        image = Image.open(osp.join(self.root,file_name)).convert('RGB')
        return image
    
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])
    

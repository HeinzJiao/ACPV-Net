#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset builder utilities.

This module resolves dataset factories from `DatasetCatalog`, attaches the
appropriate transform pipeline, and returns the corresponding dataloaders.
"""

from re import I
import torch
from .transforms import *
from . import train_dataset
from . import test_dataset
from . import vertexheatmap_dataset
from . import vertexheatmap_afm_dataset
from . import latent_vertexheatmap_dataset
from . import latent_vertexheatmap_dualseg_dataset
from . import latent_vertexheatmap_dualseg_jointlatent_dataset
from topomapper.config.paths_catalog import DatasetCatalog


def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)
    factory_name = dargs['factory']
    args = dargs['args']

    # ==== Configure transforms ====
    if (factory_name == "VertexHeatmapTrainDataset"
            or factory_name == "VertexHeatmapAfmTrainDataset"
            or factory_name == "LatentVertexHeatmapTrainDataset"
            or factory_name == "LatentVertexHeatmapDualSegTrainDataset"
            or factory_name == "LatentVertexHeatmapDualSegJointLatentTrainDataset"):
        transform = Compose([
            # Resize the image to (HEIGHT, WIDTH) if necessary.
            # This transform does not affect the associated target (e.g., mask or heatmap).
            # If the input image already has the target size, resizing will be skipped.
            ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                        cfg.DATASETS.IMAGE.WIDTH),
            # Convert both the image and annotations to torch tensors.
            ToTensor(),
            # Normalize the image tensor.
            Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                      cfg.DATASETS.IMAGE.PIXEL_STD,
                      cfg.DATASETS.IMAGE.TO_255)
        ])
    else:
        transform = Compose([
            Resize(cfg.DATASETS.IMAGE.HEIGHT,
                   cfg.DATASETS.IMAGE.WIDTH,
                   cfg.DATASETS.TARGET.HEIGHT,
                   cfg.DATASETS.TARGET.WIDTH),
            ToTensor(),
            Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                      cfg.DATASETS.IMAGE.PIXEL_STD,
                      cfg.DATASETS.IMAGE.TO_255),
        ])

    args['transform'] = transform
    args['augmentations'] = cfg.DATASETS.AUGMENTATIONS

    # ==== Build the dataset and dataloader ====
    if factory_name == "VertexHeatmapTrainDataset":
        factory = getattr(vertexheatmap_dataset, factory_name)
        collate = vertexheatmap_dataset.collate_fn
    elif factory_name == "LatentVertexHeatmapTrainDataset":
        factory = getattr(latent_vertexheatmap_dataset, factory_name)
        collate = latent_vertexheatmap_dataset.collate_fn
    elif factory_name == "LatentVertexHeatmapDualSegTrainDataset":
        factory = getattr(latent_vertexheatmap_dualseg_dataset, factory_name)
        collate = latent_vertexheatmap_dualseg_dataset.collate_fn
    elif factory_name == "LatentVertexHeatmapDualSegJointLatentTrainDataset":
        factory = getattr(latent_vertexheatmap_dualseg_jointlatent_dataset, factory_name)
        collate = latent_vertexheatmap_dualseg_jointlatent_dataset.collate_fn
    elif factory_name == "VertexHeatmapAfmTrainDataset":
        factory = getattr(vertexheatmap_afm_dataset, factory_name)
        collate = vertexheatmap_afm_dataset.collate_fn
    else:
        factory = getattr(train_dataset, factory_name)
        collate = train_dataset.collate_fn

    dataset = factory(**args)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        collate_fn=collate,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=True,
    )


def build_test_dataset(cfg):
    name = cfg.DATASETS.TEST[0]
    dargs = DatasetCatalog.get(name)
    factory_name = dargs['factory']
    args = dargs['args']

    # ==== Configure transforms ====
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255)
         ]
    )

    args['transform'] = transforms

    print("factory_name: ", factory_name)

    # ==== Build the dataset and dataloader ====
    if factory_name == "VertexHeatmapTestDataset":
        factory = getattr(vertexheatmap_dataset, factory_name)
        collate = vertexheatmap_dataset.collate_fn
    elif factory_name == "LatentVertexHeatmapTestDataset":
        factory = getattr(latent_vertexheatmap_dataset, factory_name)
        collate = latent_vertexheatmap_dataset.collate_fn
    elif factory_name == "LatentVertexHeatmapDualSegTestDataset":
        factory = getattr(latent_vertexheatmap_dualseg_dataset, factory_name)
        collate = latent_vertexheatmap_dualseg_dataset.collate_fn
    elif factory_name == "LatentVertexHeatmapDualSegJointLatentTestDataset":
        factory = getattr(latent_vertexheatmap_dualseg_jointlatent_dataset, factory_name)
        collate = latent_vertexheatmap_dualseg_jointlatent_dataset.collate_fn
    elif factory_name == "VertexHeatmapAfmTestDataset":
        factory = getattr(vertexheatmap_afm_dataset, factory_name)
        collate = vertexheatmap_afm_dataset.collate_fn
    else:
        factory = getattr(test_dataset, factory_name)
        collate = test_dataset.collate_fn

    dataset = factory(**args)
    dataset = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        collate_fn=collate,
        shuffle=False,  # No shuffling is needed during testing.
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return dataset


def build_train_dataset_multi(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset, dargs['factory'])
    args = dargs['args']
    args['transform'] = Compose(
        [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                cfg.DATASETS.IMAGE.WIDTH,
                cfg.DATASETS.TARGET.HEIGHT,
                cfg.DATASETS.TARGET.WIDTH),
         ToTensor(),
         Color_jitter(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255),
         ])
    args['rotate_f'] = cfg.DATASETS.ROTATE_F
    dataset = factory(**args)
    return dataset

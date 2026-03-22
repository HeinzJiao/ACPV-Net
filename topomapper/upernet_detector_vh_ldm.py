#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPerNet-based detector with an LDM-based latent vertex heatmap branch only.

This variant removes the semantic mask branch and keeps only the diffusion
path for latent vertex heatmap modeling conditioned on UPerNet features.
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
from topomapper.fcn_head_feat import FCNHead
from mmcv.cnn import ConvModule
from torch.utils.data.dataloader import default_collate
import numpy as np
from functools import partial
from .modules.diffusionmodules.util import extract_into_tensor, instantiate_from_config, make_beta_schedule
from .ddim import DDIMSampler
import os
import tqdm

class BuildingUPerNetDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingUPerNetDetector, self).__init__()

        # --- Backbone ---
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME

        self.num_classes = cfg.DATASETS.NUM_CLASSES  # Supports multi-class masks.

        # --- Annotation formatter ---
        if not test:
            self.formatter = AnnotationFormatter(cfg)

        # --- Main decoder head (UPerHead) ---
        decode_cfg = cfg.MODEL.DECODE_HEAD
        self.uper_head = UPerHead(
            in_channels=decode_cfg.IN_CHANNELS,  # e.g., [96, 192, 384, 768]
            in_index=decode_cfg.IN_INDEX,  # e.g., [0, 1, 2, 3]
            pool_scales=decode_cfg.POOL_SCALES,  # e.g., [1, 2, 3, 6]
            channels=decode_cfg.CHANNELS,  # e.g., 512
            dropout_ratio=decode_cfg.DROPOUT_RATIO,  # e.g., 0.1
            num_classes=self.num_classes,  # Usually read from the outer config.
            align_corners=decode_cfg.ALIGN_CORNERS  # e.g., False
        )
        self.channels = decode_cfg.CHANNELS

        # --- Feature refinement block ---
        self.jloc_head = self._make_conv(self.channels, self.channels, self.channels)

        # --- DDPM beta schedule ---
        # These values follow the original LDPoly latent diffusion setup.
        self.num_timesteps = 1000
        self.v_posterior = 0.
        self.register_schedule(beta_schedule='linear', timesteps=self.num_timesteps,
                               linear_start=0.0015, linear_end=0.0155)

        # --- DDPM denoising UNet ---
        unet_config = cfg.MODEL.unet_config
        conditioning_key = cfg.MODEL.conditioning_key
        self.model = DiffusionWrapper(diff_model_config=unet_config,
                                      conditioning_key=conditioning_key,
                                      in_channels=3,
                                      out_channels=unet_config.params.model_channels)

        # --- Inference sampling setup ---
        self.sampler = cfg.get("SAMPLER", 'direct')
        self.ddim_steps = cfg.get("DDIM_STEPS", 200)

        # --- Training step counter ---
        self.train_step = 0

    def forward(self, images, annotations=None, iteration=None, output_dir=None):
        if self.training:
            return self.forward_train(images, annotations=annotations, iteration=iteration, output_dir=output_dir)
        else:
            return self.forward_test(images, annotations=annotations, sampler=self.sampler, ddim_steps=self.ddim_steps)

    def forward_train(self, images, annotations=None, iteration=None, output_dir=None):
        """
        Training forward pass for the latent vertex heatmap diffusion branch.

        This variant does not include semantic segmentation or auxiliary
        segmentation heads.
        """
        self.train_step += 1

        # Backbone and decoder features used as the diffusion condition.
        features = self.uper_head(self.backbone(images))  # (B, C, H/4, W/4)
        cond = self.jloc_head(features)  # (B, C, H/4, W/4)

        # Ground-truth latent heatmap, shape (B, 3, H/4, W/4).
        targets, metas = self.formatter(annotations)
        latent_gt = targets['vertex_heatmap_latent']

        # Randomly sample diffusion timesteps.
        B = latent_gt.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=latent_gt.device).long()

        # Add noise with q_sample(x0, t, noise).
        noise = torch.randn_like(latent_gt)
        latent_noisy = self.q_sample(x_start=latent_gt, t=t, noise=noise)

        # Predict the corresponding noise residual.
        pred_noise = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)

        # Periodically visualize intermediate diffusion states.
        if iteration is not None and iteration % 5000 == 0:
            latent_recon = self.predict_start_from_noise(latent_noisy, t, pred_noise)
            with torch.no_grad():
                visualize(output_dir=output_dir, latent_noisy=latent_noisy, latent_gt=latent_gt,
                          latent_recon=latent_recon, images=images, iteration=iteration, t=t)

        # Compute the diffusion loss only.
        loss_dict = {}

        # L1 loss between predicted noise and ground truth noise
        loss_l1 = F.l1_loss(pred_noise, noise, reduction='none').mean(dim=(1, 2, 3))  # [B]
        loss_dict['loss_vertex_heatmap_l1'] = loss_l1.mean()

        extra_info = {}
        return loss_dict, extra_info

    def apply_model(self, x_noisy, t, cond):
        """
        Use the diffusion wrapper to predict the denoising target.

        Args:
            x_noisy: Noisy latent tensor
            t: Current timestep (int or tensor)
            cond: Conditioning feature map

        Returns:
            x_recon: Model prediction, typically the noise residual or x0
        """
        x_recon = self.model(x_noisy, t, cond=cond)
        return x_recon

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def register_schedule(self,
                          given_betas=None,
                          beta_schedule="linear",
                          timesteps=1000,
                          linear_start=1e-4,
                          linear_end=2e-2,
                          cosine_s=8e-3):
        """
        Register beta, alpha, and posterior schedules for diffusion.

        Args:
            given_betas: Custom beta sequence, if provided
            beta_schedule: Schedule type, e.g. "linear" or "cosine"
            timesteps: Number of diffusion timesteps
            linear_start / end: Beta range for the linear schedule
            cosine_s: Smoothing factor for the cosine schedule
        """
        # --- Beta schedule ---
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # --- Common diffusion buffers for q(x_t | x_{t-1}) and related terms ---
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1.)))

        # --- Posterior q(x_{t-1} | x_t, x_0) parameters ---
        posterior_variance = ((1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
                              + self.v_posterior * betas)  # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(
            posterior_variance
        ))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))
        ))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        ))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
        ))

    def q_sample(self, x_start, t, noise):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

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

    def forward_test(self, images, annotations=None, sampler="direct", ddim_steps=200):
        # Backbone and decoder features used as the diffusion condition.
        features = self.uper_head(self.backbone(images))  # (B, C, H/4, W/4)
        cond = self.jloc_head(features)  # (B, C, H/4, W/4)

        # Ground-truth latent heatmap, shape (B, 3, H/4, W/4).
        targets, metas = self.formatter(annotations)
        latent_gt = targets['vertex_heatmap_latent']

        if sampler == "direct":
            noise_x = torch.randn_like(latent_gt)
            final_t = torch.tensor([self.num_timesteps - 1], device=cond.device).long()
            pred_noise = self.apply_model(x_noisy=noise_x, t=final_t, cond=cond)
            latent_recon = self.predict_start_from_noise(noise_x, final_t, pred_noise)  # (B, 3, H/4, W/4)
        elif sampler == "ddim":
            shape = latent_gt.shape  # (B, 3, H/4, W/4)
            samples, intermediates = self.sample_log(cond=cond, shape=shape, ddim_steps=ddim_steps)
            latent_recon = samples

        output = {
            "vertex_heatmap_latent_recon": latent_recon
        }
        extra_info = {}

        return output, extra_info

    def sample_log(self, cond, shape, ddim_steps):
        ddim_sampler = DDIMSampler(self)
        samples, intermediates = ddim_sampler.sample(
            S=ddim_steps,
            shape=shape,
            conditioning=cond,
            verbose=False
        )
        return samples, intermediates

class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key, in_channels=3, out_channels=224):
        super().__init__()
        assert conditioning_key in ['concat', 'crossattn'], "Only 'concat' and 'crossattn' modes are supported"
        self.conditioning_key = conditioning_key

        # Instantiate the diffusion model.
        self.diffusion_model = instantiate_from_config(diff_model_config)

        # Channel projection used in concat-based conditioning.
        if self.conditioning_key == 'concat':
            self.latent_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t, cond=None):
        if self.conditioning_key == 'concat':
            assert cond.shape[2:] == x.shape[2:], f"Shape mismatch between x {x.shape} and cond {cond.shape}"
            x_proj = self.latent_conv(x)                  # Project the latent to the target channel dimension.
            x_concat = torch.cat([x_proj, cond], dim=1)   # Concatenate the projected latent and condition.
            return self.diffusion_model(x_concat, t)

        elif self.conditioning_key == 'crossattn':
            return self.diffusion_model(x, t, context=cond)

        else:
            raise NotImplementedError(f"Unsupported conditioning_key: {self.conditioning_key}")


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
            'vertex_heatmap_latent': ann['vertex_heatmap_latent'],
        }

        meta = {
        }

        return target, meta

def visualize(output_dir, latent_noisy, latent_gt, latent_recon, images, iteration, t):
    """
    Visualize latent maps and input images for debugging.

    Args:
        output_dir (str): Output root directory
        latent_noisy (Tensor): [B, 3, H, W]
        latent_gt (Tensor): [B, 3, H, W]
        latent_recon (Tensor): [B, 3, H, W]
        images (Tensor): [B, 3, H, W], normalized input images
        iteration (int): Current iteration, used in output file names
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "latent"), exist_ok=True)

    # Use the first sample in the batch for visualization.
    latent_gt_np = latent_gt[0].detach().cpu().numpy()        # [3, H, W]
    latent_recon_np = latent_recon[0].detach().cpu().numpy()  # [3, H, W]
    latent_noisy_np = latent_noisy[0].detach().cpu().numpy()  # [3, H, W]
    image_np = images[0].detach().cpu().numpy()               # [3, H, W]
    t = t[0].detach().cpu().numpy()

    # ----------- Latent heatmap visualization -----------
    for c in range(latent_gt_np.shape[0]):
        # Inline normalization to uint8 for visualization.
        def to_uint8(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min < 1e-5:
                return np.zeros_like(arr, dtype=np.uint8)
            arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
            return arr.astype(np.uint8)

        # Ground-truth latent.
        heatmap_gt = to_uint8(latent_gt_np[c])
        color_gt = cv2.applyColorMap(heatmap_gt, cv2.COLORMAP_VIRIDIS)
        save_path_gt = os.path.join(output_dir, "latent", f'latent_gt_{iteration}_ch{c}.png')
        cv2.imwrite(save_path_gt, color_gt)

        # Reconstructed latent.
        heatmap_recon = to_uint8(latent_recon_np[c])
        color_recon = cv2.applyColorMap(heatmap_recon, cv2.COLORMAP_VIRIDIS)
        save_path_recon = os.path.join(output_dir, "latent", f'latent_recon_{iteration}_ch{c}_t{str(t)}.png')
        cv2.imwrite(save_path_recon, color_recon)

        # Noisy latent.
        heatmap_noisy = to_uint8(latent_noisy_np[c])
        color_noisy = cv2.applyColorMap(heatmap_noisy, cv2.COLORMAP_VIRIDIS)
        save_path_noisy = os.path.join(output_dir, "latent", f'latent_noisy_{iteration}_ch{c}_t{str(t)}.png')
        cv2.imwrite(save_path_noisy, color_noisy)

    # ----------- Input image visualization (de-normalized) -----------
    pixel_mean = np.array([109.730, 103.832, 98.681]).reshape(3, 1, 1)
    pixel_std  = np.array([22.275, 22.124, 23.229]).reshape(3, 1, 1)

    image_denorm = image_np * pixel_std + pixel_mean           # De-normalize.
    image_denorm = np.clip(image_denorm, 0, 255).astype(np.uint8)
    image_vis = np.transpose(image_denorm, (1, 2, 0))          # CHW -> HWC for OpenCV.
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)     # Convert to BGR before saving.

    save_path_img = os.path.join(output_dir, "image", f'image_{iteration}.png')
    cv2.imwrite(save_path_img, image_vis)

    # ----------- Log message -----------
    print(f"[Visualize] Saved visualization for iteration {iteration} to: {output_dir}")


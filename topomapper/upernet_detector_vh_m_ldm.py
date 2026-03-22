#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPerNet-based detector with a semantic segmentation branch and an LDM-based
latent vertex heatmap branch.

The model combines:
- a backbone + UPerNet decoder for semantic mask prediction
- an auxiliary segmentation head
- a diffusion model that reconstructs latent vertex heatmaps conditioned on
  UPerNet features
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
from .modules.ema import LitEma
from contextlib import contextmanager

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

        # --- Auxiliary branch (FCNHead) ---
        aux_cfg = cfg.MODEL.AUX_HEAD
        self.auxiliary_head = FCNHead(
            in_channels=aux_cfg.IN_CHANNELS,  # e.g., 384
            in_index=aux_cfg.IN_INDEX,  # e.g., 2
            channels=aux_cfg.CHANNELS,  # e.g., 256
            num_convs=aux_cfg.NUM_CONVS,  # e.g., 1
            concat_input=aux_cfg.CONCAT_INPUT,  # e.g., False
            dropout_ratio=aux_cfg.DROPOUT_RATIO,  # e.g., 0.1
            num_classes=aux_cfg.NUM_CLASSES,  # e.g., 7
            align_corners=aux_cfg.ALIGN_CORNERS  # e.g., False
        )
        if aux_cfg.DROPOUT_RATIO > 0:
            self.dropout = nn.Dropout2d(aux_cfg.DROPOUT_RATIO)
        else:
            self.dropout = nn.Identity()

        # --- Feature refinement blocks ---
        self.mask_head = self._make_conv(self.channels, self.channels, self.channels)
        self.jloc_head = self._make_conv(self.channels, self.channels, self.channels)

        # --- Main segmentation predictor ---
        self.mask_predictor = self._make_predictor(self.channels, self.num_classes)

        # --- Auxiliary segmentation predictor ---
        self.mask_aux_predictor = nn.Sequential(
            self.dropout,
            nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(aux_cfg.CHANNELS, aux_cfg.NUM_CLASSES, kernel_size=3, padding=1)
        )

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

        # --- EMA configuration ---
        self.use_ema = getattr(cfg, "USE_EMA", False)
        self.ema_decay = getattr(cfg, "EMA_DECAY", 0.9999)
        self.ema_use_num_updates = getattr(cfg, "EMA_USE_NUM_UPDATES", True)

        if self.use_ema:
            # EMA is applied to the LDM denoiser only.
            self.model_ema = LitEma(self.model,
                                    decay=self.ema_decay,
                                    use_num_upates=self.ema_use_num_updates)

        # --- Inference sampling setup ---
        self.sampler = cfg.get("SAMPLER", 'direct')
        self.ddim_steps = cfg.get("DDIM_STEPS", 200)

        # --- Loss config ---
        self.mask_loss_type = cfg.MODEL.LOSS.MASK_LOSS_TYPE.lower()
        self.mask_aux_loss_type = cfg.MODEL.LOSS.MASK_AUX_LOSS_TYPE.lower()
        self.w_bg = float(cfg.MODEL.LOSS.W_BG)
        self.w_obj = float(cfg.MODEL.LOSS.W_OBJ)

        # --- Training step counter ---
        self.train_step = 0

    @contextmanager
    def ema_scope(self, context: str = None):
        """Temporarily switch `self.model` to EMA weights and restore afterward."""
        if getattr(self, "use_ema", False):
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield
        finally:
            if getattr(self, "use_ema", False):
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        """Update EMA after optimizer.step() using the latest online weights."""
        if getattr(self, "use_ema", False):
            # Equivalent to `self.model_ema(self.model)` in Stable Diffusion.
            self.model_ema(self.model)

    def forward(self, images, annotations=None, iteration=None, output_dir=None):
        if self.training:
            return self.forward_train(images, annotations=annotations, iteration=iteration, output_dir=output_dir)
        else:
            return self.forward_test(images, annotations=annotations, sampler=self.sampler, ddim_steps=self.ddim_steps)

    def forward_train(self, images, annotations=None, iteration=None, output_dir=None):
        """
        Training forward pass.

        The training pipeline contains three branches:
            1. Main mask branch for semantic segmentation
            2. Auxiliary mask branch for extra supervision
            3. Latent vertex heatmap branch modeled with diffusion

        Args:
            images: Input images of shape (B, 3, H, W)
            annotations: Ground truth containing mask and latent heatmap targets
            iteration: Current iteration, used for periodic visualization
            output_dir: Directory for saving visualization results

        Returns:
            loss_dict: Per-task losses for the main mask branch, auxiliary
                mask branch, and LDM branch
            extra_info: Reserved field, currently empty
        """
        self.train_step += 1

        # === Backbone and decoder features ===
        inputs = self.backbone(images)  # Multi-scale backbone features.
        features = self.uper_head(inputs)  # Main decoder features, (B, 512, H/4, W/4).
        features_aux = self.auxiliary_head(inputs)  # Auxiliary features, (B, 256, H/4, W/4).

        # === Main mask branch ===
        # mask_feature = self.mask_head(features)  # Optional mask-specific feature refinement.
        mask_feature = features
        mask_pred = self.mask_predictor(mask_feature)  # Segmentation logits.
        mask_pred = F.interpolate(mask_pred, scale_factor=4, mode='bilinear', align_corners=False)  # (B, num_classes, H, W)

        # === Auxiliary mask branch ===
        mask_pred_aux = self.mask_aux_predictor(features_aux)  # Auxiliary logits, (B, num_classes, H, W)

        # === Latent vertex heatmap branch (LDM) ===
        # cond = self.jloc_head(features)  # Optional diffusion conditioning features.
        cond = features

        # Prepare the latent ground truth and random diffusion timesteps.
        targets, metas = self.formatter(annotations)
        latent_gt = targets['vertex_heatmap_latent']
        B = latent_gt.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=latent_gt.device).long()

        # Add noise and predict the corresponding noise residual.
        latent_noisy, pred_noise, noise = self.predict_vertex_latent_heatmap_noise(latent_gt, cond, t)

        # === Losses ===
        loss_dict = self.compute_vertex_heatmap_loss(pred_noise, noise)  # Diffusion loss.

        gt_mask = targets['mask'].squeeze(1).long()
        loss_dict['loss_mask'] = self.segmentation_loss(
            mask_pred, gt_mask, loss_type=self.mask_loss_type
        )
        loss_dict['loss_mask_aux'] = self.segmentation_loss(
            mask_pred_aux, gt_mask, loss_type=self.mask_aux_loss_type
        )

        # Periodically visualize intermediate diffusion and segmentation outputs.
        if iteration is not None and iteration % 5000 == 0:
            with torch.no_grad():
                if getattr(self, "use_ema", False):
                    with self.ema_scope(context=f"viz@iter{iteration}"):
                        # Recompute pred_noise with EMA weights before reconstructing x0.
                        pred_noise_ema = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
                        latent_recon = self.predict_start_from_noise(latent_noisy, t, pred_noise_ema)
                else:
                    pred_noise_cur = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
                    latent_recon = self.predict_start_from_noise(latent_noisy, t, pred_noise_cur)

                visualize(output_dir=output_dir,
                          latent_noisy=latent_noisy, latent_gt=latent_gt, latent_recon=latent_recon,
                          mask_pred=mask_pred, mask_pred_aux=mask_pred_aux, gt_mask=gt_mask,
                          images=images, iteration=iteration, t=t)

        extra_info = {}
        return loss_dict, extra_info
    
    def segmentation_loss(self, logits, target, loss_type="ce"):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        loss_type = loss_type.lower()

        if loss_type == "ce":
            return F.cross_entropy(logits, target)

        elif loss_type == "weighted_ce":
            class_weight = torch.tensor(
                [self.w_bg, self.w_obj],
                dtype=logits.dtype,
                device=logits.device
            )
            return F.cross_entropy(logits, target, weight=class_weight)

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def predict_vertex_latent_heatmap_noise(self, latent_gt, cond, t):
        """
        Predict noise for a latent vertex heatmap at timestep `t`.

        Returns the noised latent, the predicted noise, and the sampled noise.
        """
        noise = torch.randn_like(latent_gt)
        latent_noisy = self.q_sample(x_start=latent_gt, t=t, noise=noise)
        pred_noise = self.apply_model(x_noisy=latent_noisy, t=t, cond=cond)
        return latent_noisy, pred_noise, noise

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

    def compute_vertex_heatmap_loss(self, pred_noise, noise):
        """
        Compute the diffusion loss between predicted and sampled noise.

        The current implementation uses L1 loss.
        """
        loss_l1 = F.l1_loss(pred_noise, noise, reduction='none').mean(dim=(1, 2, 3))  # shape: [B]
        return {'loss_vertex_heatmap_l1': loss_l1.mean()}

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
        # === Backbone and decoder features ===
        inputs = self.backbone(images)  # Multi-scale backbone features.
        features = self.uper_head(inputs)  # Main decoder features, (B, C, H/4, W/4).

        # === Main mask branch ===
        # mask_feature = self.mask_head(features)  # Optional mask-specific feature refinement.
        mask_feature = features
        mask_pred = self.mask_predictor(mask_feature)  # (B, num_classes, H/4, W/4)
        mask_pred = F.interpolate(mask_pred, scale_factor=4, mode='bilinear', align_corners=False)  # → (B, C, H, W)
        mask_prob = mask_pred.softmax(dim=1)  # (B, num_classes, H, W)
        seg_mask = mask_prob.argmax(dim=1)  # (B, H, W), int64

        # === Latent vertex heatmap reconstruction (LDM) ===
        # cond = self.jloc_head(features)
        cond = features

        # # Ground-truth latent heatmap would have shape (B, 3, H/4, W/4).
        # targets, metas = self.formatter(annotations)
        # latent_gt = targets['vertex_heatmap_latent']
        # We only need the latent shape: (B, 3, H/4, W/4).
        B, _, H4, W4 = features.shape
        latent_shape = (B, 3, H4, W4)

        # Use EMA weights for evaluation and inference when enabled.
        if getattr(self, "use_ema", False):
            scope = self.ema_scope(context="inference")
        else:
            # Empty context manager.
            @contextmanager
            def _null(): yield
            scope = _null()

        with scope:
            if sampler == "direct":
                # One-step prediction given random noise and final timestep
                # noise_x = torch.randn_like(latent_gt)
                noise_x = torch.randn(latent_shape, device=cond.device)
                final_t = torch.tensor([self.num_timesteps - 1], device=cond.device).long()
                pred_noise = self.apply_model(x_noisy=noise_x, t=final_t, cond=cond)
                latent_recon = self.predict_start_from_noise(noise_x, final_t, pred_noise)
            elif sampler == "ddim":
                # DDIM sampling with the same latent shape
                samples, intermediates = self.sample_log(cond=cond, shape=latent_shape, ddim_steps=ddim_steps)
                latent_recon = samples

        output = {
            "vertex_heatmap_latent_recon": latent_recon,
            "mask_prob": mask_prob,
            "seg_mask": seg_mask
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
            'mask': ann['mask']
        }

        meta = {
        }

        return target, meta

def visualize(output_dir, latent_noisy, latent_gt, latent_recon,
              mask_pred, mask_pred_aux, gt_mask, images, iteration, t):
    """
    Visualize latent maps and segmentation outputs for debugging.

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
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)

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

    # === Decode mask predictions and ground truth ===
    mask_pred_prob = F.softmax(mask_pred, dim=1)
    seg_mask_pred = mask_pred_prob.argmax(dim=1)[0].detach().cpu().numpy()  # [H, W]

    mask_pred_aux_prob = F.softmax(mask_pred_aux, dim=1)
    seg_mask_pred_aux = mask_pred_aux_prob.argmax(dim=1)[0].detach().cpu().numpy()  # [H, W]

    seg_mask_gt = gt_mask[0].detach().cpu().numpy()  # (H, W), assumes squeeze(1) has already been applied.

    def mask_to_rgb(mask):
        H, W = mask.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        if mask_pred.shape[1] == 2:
            rgb[mask == 1] = (255, 255, 255)
            return rgb

        # === Class-ID to BGR mapping ===
        id_to_color = {
            4: (237, 174, 120),  # Water       light blue
            3: (181, 226, 173),  # Vegetation  light green
            2: (234, 245, 251),  # Unvegetated off-white
            1: (193, 167, 141),  # Road        dusty blue
            0: (199, 224, 242),  # Building    warm sand
        }
        for id, color in id_to_color.items():
            rgb[mask == id] = color
        return rgb

    # Save the three mask visualizations.
    cv2.imwrite(os.path.join(output_dir, "mask", f"mask_pred_{iteration}.png"), mask_to_rgb(seg_mask_pred))
    cv2.imwrite(os.path.join(output_dir, "mask", f"mask_pred_aux_{iteration}.png"), mask_to_rgb(seg_mask_pred_aux))
    cv2.imwrite(os.path.join(output_dir, "mask", f"mask_gt_{iteration}.png"), mask_to_rgb(seg_mask_gt))

    # === Log message ===
    print(f"[Visualize] Saved latent + mask visualization for iteration {iteration} to: {output_dir}")

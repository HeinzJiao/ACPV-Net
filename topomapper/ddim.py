#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDIM sampler used for latent diffusion inference.

This module implements deterministic or stochastic DDIM sampling for the
latent diffusion models used in this repository.
"""

import torch
import numpy as np
from tqdm import tqdm
from topomapper.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    """
    DDIM Sampler for denoising latent variables.
    Supports deterministic (eta=0) and stochastic (eta>0) sampling.
    """
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model  # BuildingUperNetDetector
        self.ddpm_num_timesteps = model.num_timesteps  # 1000
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        """
        Create the DDIM sampling schedule and precompute required buffers.

        - If `ddim_eta = 0.0`, the sampling is deterministic (no noise is added).
        - If `ddim_eta > 0.0`, stochasticity is introduced in each step.
        """
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose)

        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.betas.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # Precompute common diffusion terms.
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # Compute DDIM-specific sampling parameters.
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose)

        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

        # Keep compatibility with the original DDPM step count.
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, S, shape, conditioning=None, eta=0.,
               temperature=1., noise_dropout=0., verbose=True, x_T=None,
               log_every_t=100, **kwargs):
        """
        Run full DDIM sampling from pure noise or given x_T.
        Args:
            S: Number of DDIM steps.
            shape: Full shape tuple (B, C, H, W)
            conditioning: Conditioning input.
            eta: Controls deterministic (0.0) vs stochastic (>0.0) sampling.
            x_T: Optional starting noise tensor

        Returns:
            x_0: Final denoised output.
            intermediates: Dictionary of intermediate states.
        """
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        samples, intermediates = self.ddim_sampling(
            cond=conditioning,
            shape=shape,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            x_T=x_T,
            log_every_t=log_every_t
        )
        return samples, intermediates
        # intermediates = {
        #     'x_inter': intermediate noisy latents,
        #     'pred_x0': intermediate denoised predictions
        # }

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, x_T=None, ddim_use_original_steps=False,
                      timesteps=None, log_every_t=100,
                      temperature=1., noise_dropout=0.):
        """
        Iteratively denoise starting from noise (or given x_T).
        """
        device = self.model.betas.device
        b = shape[0]
        img = x_T if x_T is not None else torch.randn(shape, device=device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            img, pred_x0 = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      temperature=temperature,
                                      noise_dropout=noise_dropout,
                                      )

            if index % log_every_t == 0 or index == total_steps - 1:  # log_every_t=100, total_steps=200
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False,
                      temperature=1., noise_dropout=0.):
        """
        Perform one DDIM denoising step.
        """
        b, *_, device = *x.shape, x.device
        e_t = self.model.apply_model(x, t, c)  # Predict noise

        # Select proper schedule
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # Select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # Estimate x0 (denoised image)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # Direction pointing to x_t
        dir_t = (1. - a_prev - sigma_t**2).sqrt() * e_t

        # Noise for stochasticity (only nonzero if eta > 0)
        noise_x = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise_x = torch.nn.functional.dropout(noise_x, p=noise_dropout)

        # Final update
        x_prev = a_prev.sqrt() * pred_x0 + dir_t + noise_x
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

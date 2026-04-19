<div align="center">
  <h1>ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery</h1>
  <p><strong>ACPV-Net</strong> is a unified framework for all-class polygonal vectorization from a single aerial image, targeting topologically consistent vector basemap generation.</p>
  <p>
    <a href="https://arxiv.org/abs/2603.16616"><img src="https://img.shields.io/badge/arXiv-2603.16616-b31b1b" alt="arXiv"></a>
    <a href="https://huggingface.co/datasets/HeinzJiao/Deventer-512"><img src="https://img.shields.io/badge/Hugging%20Face-Deventer--512-blue" alt="Hugging Face Dataset"></a>
    <a href="https://huggingface.co/HeinzJiao/deventer512_vmamba-s_m_vh-ldm_kl4_b8"><img src="https://img.shields.io/badge/Hugging%20Face-Checkpoint-orange" alt="Hugging Face Checkpoint"></a>
  </p>
</div>

## Overview

ACPV-Net combines semantically supervised multi-class segmentation, latent vertex heatmap generation, and PSLG-based topological reconstruction to convert aerial imagery into a topologically consistent vector basemap. This repository currently focuses on the `deventer512` setting.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate acpv-net
pip uninstall -y mmcv mmcv-full mmcv-lite
pip install --force-reinstall setuptools==80.9.0 wheel cython
pip install --no-build-isolation mmcv==1.7.2
cd kernels/selective_scan
pip install --no-build-isolation .
cd ../..
```

## Quick Check

```bash
python -c "import torch, timm, mmcv, yaml, cv2; print(torch.__version__, torch.version.cuda)"
python -c "import selective_scan; print('selective_scan installed')"
```

## Datasets and Checkpoints

This repository includes configurations for multiple datasets:

```bash
Deventer-512   -> config-files/deventer512_vmamba-s_m_vh-ldm_kl4_b8.yaml
WHU Building   -> config-files/whu_building_vmamba-small_512_vh_m_ldm_kl4_b8.yaml
Shanghai       -> config-files/shanghai_building_vmamba-s_m_vh-ldm_kl4_b8.yaml
```

The public release currently provides the codebase together with the [Deventer-512 benchmark](https://huggingface.co/datasets/HeinzJiao/Deventer-512) and the [Deventer-512 checkpoint](https://huggingface.co/HeinzJiao/deventer512_vmamba-s_m_vh-ldm_kl4_b8).

After downloading the dataset, place the entire `deventer_512` folder under:

```bash
data/deventer_512
```

After downloading the checkpoint, place the entire `deventer512_vmamba-s_m_vh-ldm_kl4_b8` folder under:

```bash
outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8
```

All commands below use experiments on `deventer_512` as the running example.

## Required Pretrained Weights

This project depends on pretrained models that are **not included** in this repository.

---

### 1. Autoencoder (kl-f4)

This project uses the **Stable Diffusion AutoencoderKL (kl-f4)** for latent encoding and decoding.

You can find the pretrained models in the  
[Stable Diffusion Model Zoo](https://github.com/pesser/stable-diffusion).

After downloading the **[kl-f4 VAE (f = 4, KL)](https://ommer-lab.com/files/latent-diffusion/kl-f4.zip)**, place it at:

```bash
models/first_stage_models/kl-f4/model.ckpt
```

---

### 2. VMamba Backbone (VSSM-Small)

We use the VMamba-S (VSSM-Small, s2l15) backbone pretrained on ImageNet-1K from [VMamba](https://github.com/MzeroMiko/VMamba)

After downloading the corresponding [checkpoint](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth), place it at:

```bash
pretrained/vssm_small_0229_ckpt_epoch_222.pth
```

This version is used in our experiments for efficiency considerations. Other stronger VMamba variants can also be used as backbones if needed.

## Offline Data Preprocessing

Before training the latent vertex heatmap models, preprocess the training split offline with the scripts in `data/`. The recommended workflow is:

1. Generate vertex heatmaps from vertex JSON files.
2. Apply local D4 augmentation.
3. Encode the augmented heatmaps into latent tensors.

This repository expects precomputed latent heatmaps under `train/heatmap_augmented_latent_kl-4`. Preparing them offline avoids repeated on-the-fly preprocessing and improves training efficiency.

For the detailed preprocessing steps and script usage, see [data/README.md](data/README.md).

## Training

```bash
PYTHONPATH=./:$PYTHONPATH python scripts/train_iter.py \
  --config-file config-files/deventer512_vmamba-s_m_vh-ldm_kl4_b8.yaml
```

Resume training from a checkpoint:

```bash
PYTHONPATH=./:$PYTHONPATH python scripts/train_iter.py \
  --config-file config-files/deventer512_vmamba-s_m_vh-ldm_kl4_b8.yaml \
  --resume \
  --resume-checkpoint model_iter_xxxxxxx.pth
```

## Inference and Topological Reconstruction

1. Generate semantic masks and latent vertex heatmaps:

```bash
PYTHONPATH=./:$PYTHONPATH python scripts/test.py \
  --config-file config-files/deventer512_vmamba-s_m_vh-ldm_kl4_b8.yaml \
  --sampler ddim
```

2. Decode latent vertex heatmaps back to pixel space:

```bash
PYTHONPATH=./:$PYTHONPATH python latent_decoder_accelerated.py \
  --config config-files/autoencoder_kl_f4.yaml \
  --latent_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/vertex_heatmap_latent_pt \
  --npy_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/vertex_heatmap_recon \
  --output_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/vertex_heatmap_recon_vis \
  --scale_factor 0.086164 \
  --type heatmap \
  --batch-size 8 \
  --num-workers 8
```

3. Extract vertices from reconstructed heatmaps:

```bash
python tools/extract_vertices_from_heatmap.py \
  --input_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/vertex_heatmap_recon \
  --save_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/vertices \
  --threshold 0.5 \
  --topk 1000 \
  --kernel_size 3 \
  --dist_thresh 5.0
```

4. Reconstruct polygons via PSLG-based topological reconstruction:

```bash
python tools/polygonize_pslg_batch.py \
  --seg_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/seg_mask_npy \
  --junction_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/vertices \
  --categories 0 1 2 3 4 \
  --cats_out_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/categories \
  --vis_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/vis \
  --mode ours \
  --dist_thresh 5.0 \
  --corner_eps 2.0 \
  --vss \
  --dp_fix
```

## Evaluation

Global topological consistency:

```bash
python scripts/eval_overlap_gap_share_self.py \
  --pred_dir ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/categories \
  --width 512 \
  --height 512
```

Semantic fidelity, vertex efficiency, and geometric accuracy:

```bash
PYTHONPATH=./:$PYTHONPATH python scripts/evaluation.py \
  --gt-file ./data/deventer_512/test/annotations/road.json \
  --dt-file ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/categories/road.json \
  --eval-type ciou
```

Use `--eval-type polis` or `--eval-type angle` for additional geometry metrics.

Topological fidelity:

```bash
python scripts/eval_apls.py \
  --dt_file ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/categories/road.json \
  --gt_file ./data/deventer_512/test/annotations/road.json \
  --image_folder ./data/deventer_512/test/images \
  --gt_folder ./data/deventer_512/test/network_graphs/road \
  --output_folder ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/network_graphs/road

python scripts/eval_betti_errors.py \
  --gt_file ./data/deventer_512/test/annotations/road.json \
  --dt_file ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/categories/road.json
```

## Related Research

This repository belongs to a broader research line on polygonal vectorization from aerial imagery and large-scale topographic map generation. If ACPV-Net is relevant to your work, you may also want to follow the companion papers below.

| Paper | Venue | Focus | Resources |
| --- | --- | --- | --- |
| **LDPoly** | ISPRS 2025 | Latent diffusion for polygonal road outline extraction in topographic mapping. | [Paper](https://doi.org/10.1016/j.isprsjprs.2025.10.005) · [Code](https://github.com/HeinzJiao/LDPoly) · [Data and Weights](https://drive.google.com/drive/folders/1jsjuZxFdU9a8q-m0TNCj1MfX9rixTYJl?usp=sharing) · [Demo](https://colab.research.google.com/drive/1IW5AGfn3w3y9wSquYgXolGhcVwIWkoNd#scrollTo=eval_run) |
| **RoIPoly** | ISPRS 2025 | RoI query-based building polygon extraction with logit-guided vertex interaction. | [Paper](https://doi.org/10.1016/j.isprsjprs.2025.03.030) · [Code](https://github.com/HeinzJiao/RoIPoly) |
| **PolyR-CNN** | ISPRS 2024 | End-to-end polygonal building outline extraction with efficient RoI-based modeling. | [Paper](https://doi.org/10.1016/j.isprsjprs.2024.10.006) · [Code](https://github.com/HeinzJiao/PolyR-CNN) |

## Acknowledgements

This repository benefits from the excellent open-source contributions of [HiSup](https://github.com/SarahwXU/HiSup), [Stable Diffusion](https://github.com/pesser/stable-diffusion), [LDPoly](https://github.com/HeinzJiao/LDPoly), and [VMamba](https://github.com/MzeroMiko/VMamba). We thank the authors for their great work.

# ACPV-Net

Official repository for the paper:

**ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery**  
CVPR 2026

This repository is currently being updated and will be finalized soon.

## Environment Setup

1. Create a fresh conda environment from `environment.yml`
2. Install the VMamba selective scan kernel
3. Run ACPV-Net with `PYTHONPATH` pointing to the repository root

### 1. Create the conda environment

Linux:

```bash
conda env create -f environment.yml
conda activate acpv-net-vmamba
```

Windows PowerShell:

```powershell
conda env create -f environment.yml
conda activate acpv-net-vmamba
```

This environment is designed around the same idea used by VMamba: start from a VMamba-compatible PyTorch/CUDA stack, then add ACPV-Net-specific dependencies such as `mmcv`, `omegaconf`, `pytorch-lightning`, `pycocotools`, `Shapely`, and the latent diffusion utilities.

### 2. Install `selective_scan`

This step is important. For the VMamba backbone used in this project, the `selective_scan` kernel should be installed explicitly.

```bash
cd kernels/selective_scan
pip install .
cd ../..
```

If this step is skipped, the VMamba backbone may fall back to slower or incomplete execution paths, and in practice the backbone setup is often not reliable enough for normal project use.

### 3. Use the repository root as `PYTHONPATH`

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

You can also prefix commands inline:

```bash
PYTHONPATH=./:$PYTHONPATH python scripts/test.py --config-file config-files/deventer512_vmamba-s_m_vh-ldm_kl4_b8.yaml --sampler ddim
```

## Quick Verification

After activation, it is a good idea to verify the core imports:

```bash
python -c "import torch, timm, mmcv, yaml, cv2; print(torch.__version__)"
python -c "import selective_scan; print('selective_scan installed')"
```

If the second command fails, go back to `kernels/selective_scan` and reinstall it.

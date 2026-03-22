# ACPV-Net

Official repository for the paper:

**ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery**  
CVPR 2026

This repository is currently being updated and will be finalized soon.

# ACPV-Net

## Environment Setup

The installation path below reflects the setup that was actually validated on the server for the current mainline ACPV-Net codebase.

1. Create a fresh conda environment from `environment.yml`
2. Install `mmcv` separately
3. Install the VMamba selective scan kernel

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate acpv-net
```

### 2. Install `mmcv` separately

Run the following commands after activating the environment:

```bash
pip uninstall -y mmcv mmcv-full mmcv-lite
pip install --force-reinstall setuptools==80.9.0 wheel cython
pip install --no-build-isolation mmcv==1.7.2
```

### 3. Install `selective_scan`

This step is important. For the VMamba backbone used in this project, the `selective_scan` kernel should be installed explicitly.

```bash
cd kernels/selective_scan
pip install --no-build-isolation .
cd ../..
```

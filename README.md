# ACPV-Net

ACPV-Net is a framework for All-Class Polygonal Vectorization (ACPV), which converts a single aerial image into a topologically consistent vector basemap. It combines semantically supervised conditioning for multi-class segmentation and latent vertex heatmap generation with PSLG-based topological reconstruction. This repository currently focuses on the `deventer512` setting.

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

## Datasets and Configs

This repository includes configurations for multiple datasets:

```bash
Deventer-512   -> config-files/deventer512_vmamba-s_m_vh-ldm_kl4_b8.yaml
WHU Building   -> config-files/whu_building_vmamba-small_512_vh_m_ldm_kl4_b8.yaml
Shanghai       -> config-files/shanghai_building_vmamba-s_m_vh-ldm_kl4_b8.yaml
```

This GitHub repository currently releases the codebase only. The Deventer-512 benchmark and model checkpoints will be released through Hugging Face.

Deventer-512, the benchmark introduced in our paper, will be available at:

```bash
<Hugging Face dataset link>
```

Model checkpoints will be available at:

```bash
<Hugging Face model link>
```

After downloading the dataset, place the entire `deventer_512` folder under:

```bash
data/deventer_512
```

All commands below use experiments on `deventer_512` as the example.

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
  --gt_folder ./data/deventer_512/test/network_graphs/road_bridge \
  --output_folder ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/network_graphs/road_bridge

python scripts/eval_betti_errors.py \
  --gt_file ./data/deventer_512/test/annotations/road.json \
  --dt_file ./outputs/deventer512_vmamba-s_m_vh-ldm_kl4_b8/ddim/poly_pslg/categories/road.json
```

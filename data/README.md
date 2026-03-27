# Data Preprocessing

This directory contains the utility scripts used to prepare training data before running the latent vertex heatmap models.

## Recommended Workflow

For latent vertex heatmap training, preprocess the training split offline in the following order:

1. Generate vertex heatmaps from per-image vertex JSON files with `generate_vertex_heatmaps.py`.
2. Apply local D4 augmentation to the generated heatmaps with `augment_heatmap_d4.py`.
3. Encode each augmented heatmap folder into AutoencoderKL latent tensors with `../latent_encoder_accelerated.py`.

The purpose of this workflow is to avoid generating heatmaps and encoding latents on the fly during training, which reduces preprocessing overhead and improves training throughput.

After preprocessing, the training code expects latent files under:

```bash
data/<dataset_name>/train/heatmap_augmented_latent_kl-4/<augmentation>/z/*.pt
```

## Example Commands

1. Generate vertex heatmaps:

```bash
python data/generate_vertex_heatmaps.py \
  --json_dir ./data/<dataset_name>/train/vertices \
  --save_dir ./data/<dataset_name>/train/vertex_heatmaps_sigma-3_augmented/rot0 \
  --sigma 3
```

2. Apply D4 augmentation:

```bash
python data/augment_heatmap_d4.py \
  --input_dir ./data/<dataset_name>/train/vertex_heatmaps_sigma-3_augmented/rot0 \
  --output_dir ./data/<dataset_name>/train/vertex_heatmaps_sigma-3_augmented
```

3. Encode augmented heatmaps into latents:

```bash
for aug in rot0 rot90 rot180 rot270 flip_h flip_v flip_diag flip_anti_diag; do
  PYTHONPATH=./:$PYTHONPATH python latent_encoder_accelerated.py \
    --config config-files/autoencoder_kl_f4.yaml \
    --input ./data/<dataset_name>/train/vertex_heatmaps_sigma-3_augmented/${aug} \
    --output_dir ./data/<dataset_name>/train/heatmap_augmented_latent_kl-4/${aug} \
    --type heatmap \
    --batch-size 8 \
    --num-workers 8 \
    --scale-samples 128
done
```

## Notes

- Replace `<dataset_name>` with the actual dataset folder name (e.g., `deventer_512`).
- Replace `train/vertices` with your actual vertex JSON directory if your dataset uses a different name.
- The augmentation folder names are expected to match the D4 transform names used by the dataset loader:
  `rot0`, `rot90`, `rot180`, `rot270`, `flip_h`, `flip_v`, `flip_diag`, `flip_anti_diag`.

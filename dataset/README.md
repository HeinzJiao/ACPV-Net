# Dataset Definitions

This folder contains dataset definitions used by different model variants in
the repository.

## Current LDM-based variants

- `latent_vertexheatmap_dataset.py`
  - Used by:
    - `topomapper/upernet_detector_vh_ldm.py`
    - `topomapper/upernet_detector_vh_m_ldm.py`

- `latent_vertexheatmap_dualseg_dataset.py`
  - Used by:
    - `topomapper/upernet_detector_vh_m_ldm_dualseg.py`

- `latent_vertexheatmap_dualseg_jointlatent_dataset.py`
  - Used by:
    - `topomapper/upernet_detector_vh_m_ldm_dualseg_jointlatent.py`

## Legacy / non-LDM variants

- `train_dataset.py` and `test_dataset.py`
  - Used by:
    - `topomapper/upernet_detector.py`

- `vertexheatmap_dataset.py`
  - Used by:
    - `topomapper/upernet_detector_vertexheatmap.py`
    - `topomapper/upernet_detector_vertexheatmap_deconv.py`

- `vertexheatmap_afm_dataset.py`
  - Used by:
    - `topomapper/upernet_detector_vertexheatmap_afm.py`

## Shared utilities

- `build.py`
  - Resolves dataset factories and builds dataloaders.

- `transforms.py`
  - Provides shared preprocessing and transform classes.

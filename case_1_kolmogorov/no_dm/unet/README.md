# U-Net Reproduction for Kolmogorov Flow

This folder contains my reproduction of the U-Net neural operator and
diffusion-based refinement model for Case 1: Kolmogorov flow.

## Main files

### Neural Operator

- [`train_no.py`](train_no.py): training script for the U-Net neural operator
- [`no_postprocess.ipynb`](no_postprocess.ipynb): evaluation and visualization results
- [`MnM.py`](MnM.py): U-Net model architecture

### Diffusion Model

- [`dm/train_dm.py`](dm/train_dm.py): diffusion model training script
- [`dm/dm_postprocess.ipynb`](dm/dm_postprocess.ipynb): final comparison and visualization results

## Results

The main reproduction results are shown in:

- [Neural operator results](no_postprocess.ipynb)
- [Neural operator + diffusion model results](dm/dm_postprocess.ipynb)

Generated datasets and trained model weights are excluded from this repository
because of their large file sizes.
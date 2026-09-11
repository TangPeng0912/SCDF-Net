# SCDF-Net

Official PyTorch implementation of **SCDF-Net: Unsupervised Scale-Conditioned Hyperspectral and Multispectral Image Fusion via a Frequency-Spatial Dual-Domain Network**.

SCDF-Net is an **unsupervised** framework for hyperspectral–multispectral image (HSI–MSI) fusion that supports both **integer and fractional** spatial resolution ratios. It operates directly on the native input grids and does not require any ground-truth HR-HSI labels.

---

## Highlights

- **Two-stage training strategy**: Stage 1 self-supervised estimates the PSF and SRF from the observed LR-HSI/HR-MSI pair; Stage 2 freezes these degradation operators as physical priors to guide unsupervised fusion training.
- **Continuous scale embedding**: The spatial scale factor is treated as an explicit continuous conditioning variable, enabling fusion under arbitrary integer and fractional ratios (e.g., ×1.5, ×2.4, ×4.0, ×5.0, ×6.0).
- **Frequency–spatial dual-domain network**: A scale-conditioned frequency-domain branch and a spatial-domain branch are jointly optimized to preserve high-frequency details while maintaining spectral fidelity.
- **Fully unsupervised**: No paired HR-HSI training data required.

---

## Project Structure

```
SCDF-Net/
├── train.py                    # Stage 1 & Stage 2 training
├── test.py                     # Test script
├── dataset.py                  # Dataset class (mixed integer/fractional degradation)
├── options_anyscale.py         # Arguments / hyperparameters
├── utils.py                    # Metrics, checkpointing, plotting
├── utils_plot.py               # Metrics-vs-epoch plotting
├── models/
│   ├── dme_net_random.py       # DME-Net (degradation model estimation, Stage 1)
│   └── sffo_net.py             # SCDF-Net (SFFO_NET_DC, Stage 2)
├── data/                       # Datasets (not included)
├── checkpoints/                # Model checkpoints (not included)
├── results/                    # Output results (not included)
├── requirements.txt
└── README.md
```

---

## Environment

- Python 3.8+
- PyTorch 2.0+ (with CUDA, if GPU available)
- NumPy, SciPy, Matplotlib
- tifffile, pandas, xlrd

Tested on: Python 3.8, PyTorch 2.0.0+cu118, CUDA 11.8, NVIDIA RTX 4090D.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The datasets are **not** included in this repository. Please download them from their official sources and place them under `data/` following the structure below.

```
data/
├── paviau/REF.mat
├── houston18/REF.mat
├── wadc/REF.mat
├── Chikusei/*.tif
├── wavelength/
│   ├── paviau.txt
│   ├── houston18.txt
│   ├── wadc.txt
│   └── Chikusei.txt
└── SRF/
    └── landsat.xls
```

- Each `<dataset>/` folder should contain the reference HR-HSI (`.mat` for PaviaU / Houston18 / WADC, `.tif` for Chikusei).
- Each `wavelength/<dataset>.txt` file contains comma-separated wavelengths in nanometers.
- `SRF/landsat.xls` provides the spectral response function used to simulate the HR-MSI.

Official dataset sources:

- **Pavia University**: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- **Houston 2018**: https://hyperspectral.ee.uh.edu/?page_id=1075
- **Washington DC Mall**: https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html
- **Chikusei**: https://naotoyokoya.com/Download.html

---

## Training

The framework is trained in **two stages**. Stage 1 must be completed before running Stage 2, because Stage 2 freezes the PSF/SRF learned in Stage 1.

### Stage 1: Degradation model estimation

```bash
python train.py \
    --dataset_name paviau \
    --srf_name landsat \
    --scale_factor 2.0 \
    --stage 1 \
    --num_epochs_first_stage 10000 \
    --lr_stage1 1e-4 \
    --batch_size 1 \
    --device cuda
```

The PSF/SRF checkpoint will be saved to:

```
checkpoints/<dataset_name>_<suffix>/checkpoint_first_stage_final.pth
```

### Stage 2: Unsupervised fusion training

```bash
python train.py \
    --dataset_name paviau \
    --srf_name landsat \
    --scale_factor 2.0 \
    --stage 2 \
    --second_stage_model SFFO_NET_DC \
    --num_epochs_second_stage 10000 \
    --lr_stage2 1e-4 \
    --batch_size 1 \
    --mixed_scales 2.0 \
    --eval_scale 2.0 \
    --sc_w 0.02 --spec_w 0.015 --sam_hsi_w 0.01 \
    --freq_w 0.01 --freq_keep 0.15 \
    --device cuda
```

Notes:

- `--mixed_scales` accepts a comma-separated list (e.g., `"2.0,2.4,4.0"`) for joint multi-scale training. Each scale requires its own Stage 1 checkpoint.
- `--scale_factor` can be an **integer** (e.g., `2.0`, `4.0`) or a **fractional** (e.g., `1.5`, `2.4`, `3.6`) value.

---

## Testing

```bash
python test.py \
    --dataset_name paviau \
    --srf_name landsat \
    --scale_factor 2.0 \
    --second_stage_model SFFO_NET_DC \
    --device cuda
```

The reconstructed HR-HSI and evaluation metrics (PSNR, SAM, RMSE, ERGAS, CC) are saved to:

```
results/<dataset_name>_<suffix>/test/
```

---

## Model Overview

| Component | Description |
| :--- | :--- |
| **Stage 1 — DME-Net** | Self-supervised estimation of the spatial PSF and spectral SRF, with non-negativity and normalization constraints. |
| **Stage 2 — SCDF-Net (SFFO_NET_DC)** | Scale-conditioned frequency–spatial dual-domain fusion network with a lightweight UNet-style refiner. |
| **Scale embedding** | Maps the continuous scale vector `s = [H/h, W/w]` into a learnable embedding that modulates the gating of both branches. |
| **Losses** | `L_LV` and `L_HV` (self-supervised reconstruction), `L_SAM` (HSI-space spectral consistency), `L_FREQ` (MSI-space frequency consistency). |

Parameter count: **≈3.1M**. Inference time: **≈0.0071 s** per image on an RTX 4090D (Houston18 dataset).

---

## Evaluation Metrics

- **PSNR** — Peak Signal-to-Noise Ratio
- **SAM** — Spectral Angle Mapper
- **RMSE** — Root Mean Square Error
- **ERGAS** — Relative Dimensionless Global Error in Synthesis
- **CC** — Correlation Coefficient

---


## License

This project is released for **academic and research purposes only**.

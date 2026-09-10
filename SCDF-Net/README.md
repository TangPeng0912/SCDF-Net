# SCDF-Net

Official PyTorch implementation of **SCDF-Net: Scale-Conditioned Dual-Domain Fusion Network for Arbitrary-Scale Hyperspectral and Multispectral Image Fusion**.

## Project Structure

```
SCDF-Net/
├── train.py
├── test.py
├── models/
├── datasets/
├── configs/
├── utils/
├── scripts/
├── data/
```

## Environment

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- tifffile
- pandas
- xlrd

Install:

```bash
pip install -r requirements.txt
```

## Dataset

Download the datasets from their official sources and place them under `data/`.

Pretrained checkpoints are **not** included.

## Training

Stage 1:
```bash
python train.py --stage 1
```

Stage 2:
```bash
python train.py --stage 2
```

## Testing

```bash
python test.py
```

## Citation

If this repository is used in your research, please cite the associated paper.

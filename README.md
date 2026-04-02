# High-Resolution Optical Satellite Image Guided DEM Super-Resolution via Topographic-Aware Transformer

<p align="center">
  <a href="https://ieeexplore.ieee.org/Xplore/home.jsp"><img src="https://img.shields.io/badge/Paper-TGRS_2026-blue"></a>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
  <a href="https://github.com/yourusername/yourrepo/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.10-orange"></a>
</p>

This repository is the official PyTorch implementation of the paper **"High-Resolution Optical Satellite Image Guided DEM Super-Resolution via Topographic-Aware Transformer"** (Accepted by *IEEE Transactions on Geoscience and Remote Sensing*, 2026).


## 🛠 Prerequisites

* Operating System: Linux / Windows
* Python 3.8+


**Environment Setup:**
```bash

conda create -n SR python=3.8
conda activate SR
pip install -r requirements.txt
```

**Data Preparation:**
```bash

data/
├── train/
│   ├── lr_dem/        # LR DEM inputs (.tif)
│   ├── hr_img/    # HR Optical guidance images (.tif)
│   └── gt_dem/        # Ground Truth HR DEMs (.tif)
└── test/
    ├── lr_dem/
    ├── hr_optical/
    └── gt_dem/
data/
├── train/
│   ├── img/
│   └── dem/
└── test/
    ├── img/
    └── dem/

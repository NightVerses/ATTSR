Guided DEM Super-Resolution via Topographic-Aware Transformer
This repository is the official implementation of "High-Resolution Optical Satellite Image Guided DEM Super-Resolution via Topographic-Aware Transformer" (TGRS 2026).

🛠 Prerequisites
Python 3.8+

PyTorch >= 1.10

CUDA (Highly recommended)

pip install -r requirements.txt

🚀 Usage
1. Data Preparation
Place your dataset (Optical images and low-res DEMs) in the data/ folder following this structure:

Plaintext
data/
├── train/
│   ├── lr_dem/
│   ├── hr_optical/
│   └── gt_dem/
└── test/
    ├── lr_dem/
    └── hr_optical/
2. Training (训练)
To train the Topographic-Aware Transformer from scratch, run:

Bash
python train.py --batch_size 16 --epochs 100 --lr 1e-4 --data_path ./data/train
--guided: Enable/disable optical image guidance.

--topo_aware: Toggle topographic-aware loss functions.

3. Inference / Prediction (预测)
To perform super-resolution on your own data using a pre-trained checkpoint:

Bash
python predict.py --checkpoint ./weights/best_model.pth --input_lr ./data/test/lr_dem --input_guide ./data/test/hr_optical --output ./results
Results: The super-resolved DEMs will be saved as .tif or .npy files in the results/ directory.

📊 Evaluation
To calculate RMSE, MAE, and Topographic Fidelity:

Bash
python evaluate.py --target ./results --gt ./data/test/gt_dem

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from osgeo import gdal
from tqdm import tqdm

from models.ATTSR import ATTSR




from utils.dataset import build_test_dataloader

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


ignore_test_file = ['UY352927', 'EB643114', 'FP496452', 'HS391158', 'KQ663679', 'LB658922', 'UY352927']

dtype = torch.cuda.FloatTensor
KERNEL_TYPE = 'lanczos2'

gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')


class Slope(nn.Module):
    def __init__(self):
        super(Slope, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1
        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * 10)
        weight2 = weight2 / (8 * 10)
        self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置
    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        ij_slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        ij_slope = torch.arctan(ij_slope) * 180 / math.pi
        return ij_slope
compute_gradient_map = Slope().cuda()

class HybirdLoss(torch.nn.Module):
    def __init__(self):
        super(HybirdLoss, self).__init__()
        self.MSE = nn.MSELoss().to(DEVICE)
        self.L1 = nn.L1Loss()

    def forward(self, pred, mask):
        gradient_target = compute_gradient_map(mask)
        gradient_output = compute_gradient_map(pred)
        MSEloss = self.MSE(pred , mask)
        L1loss = self.L1(pred,mask)

        Lterrain = self.MSE(gradient_output, gradient_target)
        loss = MSEloss+L1loss+0.09*Lterrain


        return loss



def calculate_mse(array1, array2):

    mse = np.mean((array1 - array2) ** 2)
    return mse


def get_args():
    parser = argparse.ArgumentParser(description="Inference model")
    parser.add_argument('--data_root', type=str, default='./data/Austria', help='Root directory of the dataset')
    parser.add_argument('--model_path', type=str, default=r'./checkpoints/ATTSR.pth',
                        help='Path to the saved model weights (.pth)')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save inference results')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    test_img_path = os.path.join(args.data_root, 'img')
    test_dem_path = os.path.join(args.data_root, 'dem')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")

    x_data = sorted(glob.glob(os.path.join(test_img_path, '*.tif')))
    y_data = sorted(glob.glob(os.path.join(test_dem_path, '*.tif')))

    if len(x_data) == 0:
        raise FileNotFoundError(f"No .tif files found in {test_img_path}")

    test_dataset = [x_data, y_data]

    model = ATTSR(args, scale_factor=3).cuda()

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model.eval()

    test_loader = build_test_dataloader(test_dataset, eval=True)

    for image, label, path, path2 in tqdm(test_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        label_np = label.cpu().data.numpy()
        base_min = np.min(label_np)
        base_max = np.max(label_np)

        label_norm = 2 * ((label_np - base_min) / (base_max - base_min + 10)) - 1
        label_tensor = torch.tensor(label_norm).to(DEVICE)

        lr = F.interpolate(label_tensor, scale_factor=1 / 3, mode='nearest')

        bu_lr = F.interpolate(lr, scale_factor=3, mode='bicubic', align_corners=False).cpu().data.numpy()
        bu_lr = ((bu_lr + 1) / 2) * (base_max - base_min + 10) + base_min

        output, output1, output2 = model(lr, image)
        output = (output + output1 + output2) / 3

        output = output.data.cpu().numpy()
        output = ((output + 1) / 2) * (base_max - base_min + 10) + base_min

        label_orig = ((label_norm + 1) / 2) * (base_max - base_min + 10) + base_min

        for i in range(output.shape[0]):
            img = output[i]
            img_transposed = np.transpose(img, (1, 2, 0))

            dataset = gdal.Open(path2[0])
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()

            out_filename = os.path.basename(path[i])
            outpath = os.path.join(args.save_dir, out_filename)

            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(outpath, img_transposed.shape[1], img_transposed.shape[0],
                                        img_transposed.shape[2], gdal.GDT_Float32)
            out_dataset.SetGeoTransform(geotransform)
            out_dataset.SetProjection(projection)

            for j in range(img_transposed.shape[2]):
                out_dataset.GetRasterBand(j + 1).WriteArray(img_transposed[:, :, j])

            out_dataset.FlushCache()
            out_dataset = None
            dataset = None

    print(f"Inference complete. Results saved to {args.save_dir}")
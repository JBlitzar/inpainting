import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_tensor
#from torchvision.metrics import ssim
IMAGE_MAX = 1
#https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
#https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, IMAGE_MAX]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(torch.max(img1) / torch.sqrt(mse))
    

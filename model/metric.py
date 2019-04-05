import math

import torch


def PSNR(deblurred, sharp):
    """Peak Signal to Noise Ratio"""
    mse = torch.mean((deblurred - sharp) ** 2)  # mean square error
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX ** 2 / mse)

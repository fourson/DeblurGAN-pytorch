import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.util import ensure_dir


def main(blurred_dir, sharp_dir, aligned_dir):
    image_names = os.listdir(blurred_dir)  # we assume that blurred and sharp images have the same names
    ensure_dir(aligned_dir)
    for image_name in tqdm(image_names, ascii=True):
        # convert PIL image to numpy array (H, W, C)
        blurred = np.array(Image.open(os.path.join(blurred_dir, image_name)).convert('RGB'), dtype=np.uint8)
        sharp = np.array(Image.open(os.path.join(sharp_dir, image_name)).convert('RGB'), dtype=np.uint8)
        aligned = np.concatenate((blurred, sharp), axis=1)  # horizontal alignment
        Image.fromarray(aligned).save(os.path.join(aligned_dir, image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make aligned data from raw data')

    parser.add_argument('-b', '--blurred', required=True, type=str, help='dir of blurred images')
    parser.add_argument('-s', '--sharp', required=True, type=str, help='dir of sharp images')
    parser.add_argument('-a', '--aligned', required=True, type=str, help='dir to save aligned images')

    args = parser.parse_args()

    main(args.blurred, args.sharp, args.aligned)

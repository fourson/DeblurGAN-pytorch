import os
import argparse

from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch

import model.model as module_arch
from utils.util import denormalize


def main(blurred_dir, deblurred_dir, resume):
    # load checkpoint
    checkpoint = torch.load(resume)
    config = checkpoint['config']

    # build model architecture
    generator_class = getattr(module_arch, config['generator']['type'])
    generator = generator_class(**config['generator']['args'])

    # prepare model for deblurring
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    if config['n_gpu'] > 1:
        generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    # start to deblur
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
    ])
    with torch.no_grad():
        for image_name in tqdm(os.listdir(blurred_dir)):
            blurred_img = Image.open(os.path.join(blurred_dir, image_name)).convert('RGB')
            blurred = transform(blurred_img).unsqueeze(0).to(device)
            deblurred = generator(blurred)
            deblurred_img = to_pil_image(denormalize(deblurred).squeeze().cpu())
            deblurred_img.save(os.path.join(deblurred_dir, 'deblurred ' + image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblur your own image!')

    parser.add_argument('-b', '--blurred', required=True, type=str, help='dir of blurred images')
    parser.add_argument('-d', '--deblurred', required=True, type=str, help='dir to save deblurred images')
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')

    args = parser.parse_args()

    main(args.blurred, args.deblurred, args.resume)

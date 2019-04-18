import os
import argparse

from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import torch

import model.model as module_arch
import model.metric as module_metric
from data_loader.data_loader import CustomDataLoader
from utils.util import denormalize


def main(blurred_dir, deblurred_dir, resume):
    # load checkpoint
    checkpoint = torch.load(resume)
    config = checkpoint['config']

    # setup data_loader instances
    data_loader = CustomDataLoader(data_dir=blurred_dir)

    # build model architecture
    generator_class = getattr(module_arch, config['generator']['type'])
    generator = generator_class(**config['generator']['args'])

    generator.summary()

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # prepare model for deblurring
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    if config['n_gpu'] > 1:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    # start to deblur
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            blurred = sample['blurred'].to(device)
            image_name = sample['image_name']

            deblurred = generator(blurred)

            deblurred_img = to_pil_image(denormalize(deblurred).squeeze().cpu())

            deblurred_img.save(os.path.join(deblurred_dir, 'deblurred ' + image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblur your own image!')

    parser.add_argument('-b', '--blurred', required=True, type=str, help='dir of blurred images')
    parser.add_argument('-d', '--deblurred', required=True, type=str, help='dir to save deblurred images')
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.blurred, args.deblurred, args.resume)

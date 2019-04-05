import os
import argparse

import torch
from tqdm import tqdm
import numpy as np

import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch


def main(config, resume):
    # setup data_loader instances
    data_loader_class = getattr(module_data, config['data_loader']['type'])
    config['data_loader']['args']['validation_split'] = 0.0  # do not split, just use the full dataset
    data_loader = data_loader_class(**config['data_loader']['args'])

    # build model architecture
    generator_class = getattr(module_arch, config['generator']['type'])
    generator = generator_class(**config['generator']['args'])

    discriminator_class = getattr(module_arch, config['discriminator']['type'])
    discriminator = discriminator_class(**config['discriminator']['args'])

    generator.summary()
    discriminator.summary()

    # get function handles of loss and metrics
    loss_fn = {k: getattr(module_loss, v) for k, v in config['loss'].items()}
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load checkpoint
    checkpoint = torch.load(resume)

    # prepare model for testing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    if config['n_gpu'] > 1:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])

    generator.eval()
    discriminator.eval()

    total_loss = 0.0
    total_metrics = np.zeros(len(metric_fns))

    with torch.no_grad():
        # for batch_idx, sample in enumerate(data_loader):
        for batch_idx, sample in enumerate(tqdm(data_loader)):
            blurred = sample['blurred'].to(device)
            sharp = sample['sharp'].to(device)

            deblurred = generator(blurred)
            deblurred_discriminator_out = discriminator(deblurred)

            # computing loss, metrics on test set
            content_loss_lambda = config['others']['content_loss_lambda']
            adversarial_loss_fn = loss_fn['adversarial']
            content_loss_fn = loss_fn['content']
            kwargs = {
                'deblurred_discriminator_out': deblurred_discriminator_out
            }
            loss = adversarial_loss_fn('G', **kwargs) + content_loss_fn(deblurred, sharp) * content_loss_lambda

            total_loss += loss.item()
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(deblurred, sharp)

    n_samples = len(data_loader)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeblurGAN')

    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Checkpoint file need to be specified. Add '-r model_best.pth', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)

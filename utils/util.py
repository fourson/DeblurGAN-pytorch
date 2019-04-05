import os

import torch
from torchvision import transforms


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda):
    if lr_lambda == 'origin_lr_scheduler':
        # the same as origin paper's method (epoch=300)
        # "After the first 150 epochs we linearly decay the rate to zero over the next 150 epochs"
        return lambda epoch: (1 - (epoch - 150) / 150) if epoch > 150 else 1
    # add other lambdas if you want
    else:
        raise NotImplementedError('lr_lambda [%s] is not found' % lr_lambda)


def get_lr_scheduler(lr_scheduler_config, optimizer):
    lr_scheduler_class = getattr(torch.optim.lr_scheduler, lr_scheduler_config['type'])
    if lr_scheduler_config['type'] == 'LambdaLR':
        lr_lambda = get_lr_lambda(lr_scheduler_config['args']['lr_lambda'])
        return lr_scheduler_class(optimizer, lr_lambda)
    else:
        return lr_scheduler_class(optimizer, **lr_scheduler_config['args'])


def batch_denormalize(tensor, mean, std):
    # convert mean and std to tensor
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    # tensor: (N,C,H,W)
    # denormalize dims excluding the batch_dim
    for i in range(tensor.size(0)):
        tensor[i] = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(tensor[i])
    return tensor

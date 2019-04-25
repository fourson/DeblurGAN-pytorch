import functools

import torch
import torch.nn as nn
from torchvision import models

CONV3_3_IN_VGG_19 = models.vgg19(pretrained=True).features[:15].cuda()


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        # we should never set track_running_stats to True in InstanceNorm
        # because it behaves differently in training and testing mode
        norm_layer = functools.partial(nn.InstanceNorm2d, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.zeros_(m.bias)


class ResNetBlock(nn.Module):
    """ResNet block"""

    def __init__(self, dim, norm_layer, padding_type, use_dropout, use_bias):
        super(ResNetBlock, self).__init__()

        sequence = list()
        padding = self._chose_padding_type(padding_type, sequence)

        sequence += [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=padding, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def _chose_padding_type(self, padding_type, sequence):
        padding = 0
        if padding_type == 'reflect':
            sequence += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            sequence += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        return padding

    def forward(self, x):
        out = x + self.model(x)
        return out


class MinibatchDiscrimination(nn.Module):
    """minibatch discrimination"""

    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features  # A
        self.out_features = out_features  # B
        self.kernel_dims = kernel_dims  # C
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims)).cuda()  # AxBxC
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))  # NxBC
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)  # NxBxC

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0))  # NxB
        if self.mean:
            o_b /= x.size(0)

        x = torch.cat((x, o_b), 1)  # Nx(A+B)
        return x

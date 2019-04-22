import functools

import torch
import torch.nn as nn

from .layer_utils import get_norm_layer, ResNetBlock, MinibatchDiscrimination
from base.base_model import BaseModel


class ResNetGenerator(BaseModel):
    """Define a generator using ResNet"""

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_type='instance', padding_type='reflect',
                 use_dropout=True, learn_residual=True):
        super(ResNetGenerator, self).__init__()

        self.learn_residual = learn_residual

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # downsample the feature map
            mult = 2 ** i
            sequence += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        for i in range(n_blocks):  # ResNet
            sequence += [
                ResNetBlock(ngf * 2 ** n_downsampling, norm_layer, padding_type, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            mult = 2 ** (n_downsampling - i)
            sequence += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        sequence += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        if self.learn_residual:
            out = x + out
            out = torch.clamp(out, min=-1, max=1)  # clamp to [-1,1] according to normalization(mean=0.5, var=0.5)
        return out


class NLayerDiscriminator(BaseModel):
    """Define a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance', use_sigmoid=False,
                 use_minibatch_discrimination=False):
        super(NLayerDiscriminator, self).__init__()

        self.use_minibatch_discrimination = use_minibatch_discrimination

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kernel_size = 4
        padding = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=2, padding=padding,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        if self.use_minibatch_discrimination:
            out = out.view(out.size(0), -1)
            a = out.size(1)
            out = MinibatchDiscrimination(a, a, 3)(out)
        return out

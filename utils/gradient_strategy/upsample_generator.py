import numpy as np
import torch
from torchvision import transforms
from utils.gradient_strategy.strategy import Strategy
from PIL import Image
from torch.functional import F
from torch import nn


class UpSampleGenerator(Strategy):
    def __init__(self, batch_size=32, factor=4.0):
        super(UpSampleGenerator, self).__init__()
        self.batch_size = batch_size
        self.factor = factor

    def generate_ps(self, inp, num_eval, level=None):
        shape = inp.shape
        p_small = torch.randn(num_eval, shape[1], int(shape[2] / self.factor), int(shape[3] / self.factor))
        m = nn.UpsamplingBilinear2d(scale_factor=2)
        ps = m(p_small)
        return ps

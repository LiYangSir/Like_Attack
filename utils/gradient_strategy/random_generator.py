from utils.gradient_strategy.strategy import Strategy
import torch
from torch import nn
from torch.functional import F
from scipy import signal
import numpy as np


class RandomGenerator(Strategy):

    def __init__(self, constraint):
        super(RandomGenerator, self).__init__()
        self.constraint = constraint

    def generate_ps(self, inp, num_eval, level=None):
        noise_shape = [num_eval] + list(inp.shape[1:])
        if self.constraint == 'l2':
            ps = torch.randn(*noise_shape).to(self.device)
        else:
            ps = (torch.rand(*noise_shape) * 2 - 1).to(self.device)

        return ps

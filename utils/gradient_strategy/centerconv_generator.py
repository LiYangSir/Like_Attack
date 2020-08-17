from utils.gradient_strategy.strategy import Strategy
import torch
from scipy.signal import butter, filtfilt
import numpy as np
import cv2
from scipy import fftpack
from torch import nn
from torchvision import transforms
from torch.functional import F


class CenterConvGenerator(Strategy):
    def __init__(self, factor, *shape):
        super(CenterConvGenerator, self).__init__()
        self.factor = factor
        self.mask = torch.ones(shape).to(self.device)
        for i in range(shape[0]):
            for j in range(shape[0]):
                if (np.abs(i - shape[0] / 2) ** 2 + np.abs(j - shape[1] / 2) ** 2) > (shape[0] / 2) ** 2:
                    self.mask[i, j] = 0
        self.mask = self.mask > 0

    def generate_ps(self, inp, num_eval, level=0.1):
        shape = inp.shape
        h_use, w_use = int(shape[2] / self.factor), int(shape[3] / self.factor)
        p = torch.zeros(num_eval, *shape[1:])
        p_small = torch.randn(num_eval, shape[1], h_use, w_use)
        p_small = torch.where(self.mask, p_small, torch.tensor(0.).to(self.device))
        p[:, :, (shape[2] - h_use) // 2: (shape[2] + h_use) // 2, (shape[3] - w_use) // 2: (shape[3] + w_use) // 2] = p_small
        ps = self.blur(p)
        return ps

    def blur(self, tensor_image):
        kernel = [[0., 1., 1.],
                  [0., 3., 1.],
                  [0., 0., 0.]]
        kernel_1 = [[1., 0., 1.],
                    [0., 0, 0.],
                    [1., 0., 1.]]

        min_batch = tensor_image.size()[0]
        channels = tensor_image.size()[1]
        out_channel = channels
        kernel = torch.tensor(kernel_1).expand(out_channel, channels, 3, 3)
        weight = nn.Parameter(data=kernel, requires_grad=False)

        return F.conv2d(tensor_image, weight, padding=1)

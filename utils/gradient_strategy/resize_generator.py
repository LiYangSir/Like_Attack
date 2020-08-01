import numpy as np
# from skimage import transform
import torch
from torchvision import transforms
from utils.gradient_strategy.strategy import Strategy


class ResizeGenerator(Strategy):
    def __init__(self, batch_size=32, factor=4.0):
        super(ResizeGenerator, self).__init__()
        self.batch_size = batch_size
        self.factor = factor

    def generate_ps(self, inp, num_eval, level=None):
        ps = []
        for _ in range(num_eval):
            shape = inp.shape
            assert len(shape) == 4 and shape[1] == 3
            p_small = torch.randn(shape[0], shape[1], int(shape[2] / self.factor), int(shape[3] / self.factor)).to(
                self.device)
            transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.Resize(tuple(shape[2:])), transforms.ToTensor()])
            p_small = transform(p_small[0])
            ps.append(p_small)
        ps = torch.stack(ps, 0)
        return ps

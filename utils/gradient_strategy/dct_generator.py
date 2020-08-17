import numpy as np
from scipy import fftpack
import torch
from utils.gradient_strategy.strategy import Strategy


def get_2d_idct(x):
    return fftpack.idct(fftpack.idct(x.T, norm='ortho').T, norm='ortho')


def rgb_signal_idct(signal):
    # assert len(signal.shape) == 3 and signal.shape[0] == 3
    img = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        img[c] = get_2d_idct(signal[c])
    return img


class DCTGenerator(Strategy):
    def __init__(self, factor, batch_size=32):
        super(DCTGenerator, self).__init__()
        self.factor = factor
        self.batch_size = batch_size

    def generate_ps(self, inp, num_eval, level=None):

        inp = inp[0].cpu().numpy()
        channel, height, width = inp.shape
        h_use, w_use = int(height / self.factor), int(width / self.factor)

        ps = []
        for _ in range(num_eval):
            p_signal = np.zeros_like(inp)
            for c in range(channel):
                rv = np.random.randn(h_use, w_use)
                rv_ortho, _ = np.linalg.qr(rv, mode='full')
                p_signal[c, :h_use, :w_use] = rv_ortho
            p_img = rgb_signal_idct(p_signal)
            ps.append(p_img)
        ps = np.stack(ps, axis=0)
        ps = torch.from_numpy(ps).to(self.device)
        return ps

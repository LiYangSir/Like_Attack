import abc
import torch


class Strategy(metaclass=abc.ABCMeta):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def generate_ps(self, inp, num_eval, level=None):
        pass

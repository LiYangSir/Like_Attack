import torch
import os
from config.config import model_path
import models
import threading
from concurrent.futures import ThreadPoolExecutor
from torchvision.models.densenet import _load_state_dict
import re


def densenet_load(state_dict):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


class ImageModel:

    def __init__(self, model_name, dataset_name, suffix='pth'):
        url = '{}/{}_{}.{}'.format(model_path, model_name, dataset_name, suffix)
        assert os.path.exists(url)
        self.model_name = model_name
        print("+-------Generate NetWork--------+")
        self.model = models.__dict__['{}_{}'.format(model_name, dataset_name)]()
        print("+---------Load NetWork----------+")
        state_dict = torch.load(url, map_location=torch.device('cpu'))
        if model_name.startswith('densenet'):
            state_dict = densenet_load(state_dict)

        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.pool = ThreadPoolExecutor(max_workers=10)

    def predict(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        results = torch.zeros(x.shape[0])
        x_split = torch.split(x, 30, 0)
        res = [None] * len(x_split)

        for i, x in enumerate(x_split):
            res[i] = self.pool.submit(self._predict, x)
        for i, a in enumerate(res):
            result = a.result()
            results[i * 30: i * 30 + result.shape[0]] = result
        return results

    def _predict(self, x):
        pred = self.model(x)
        pred = torch.argmax(pred, 1)
        return pred

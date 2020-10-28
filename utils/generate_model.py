import torch
import os
from config.config import model_path
import models
import time
from concurrent.futures import ThreadPoolExecutor
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def __init__(self, model_name, dataset_name, suffix='pth', thread_pool=False):
        url = '{}/{}_{}.{}'.format(model_path, model_name, dataset_name, suffix)
        assert os.path.exists(url)
        self.model_name = model_name
        print("+-------Generate NetWork--------+")
        self.model = models.__dict__['{}_{}'.format(model_name, dataset_name)]()
        print("+---------Load NetWork----------+")
        state_dict = torch.load(url, map_location=device)
        if model_name.startswith('densenet169'):
            state_dict = densenet_load(state_dict)
        self.thread_pool = thread_pool
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        if thread_pool:
            self.pool = ThreadPoolExecutor(max_workers=10)

    def predict(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        x_split = torch.split(x, 30, 0)

        if not self.thread_pool:
            results = [self._predict(image) for image in x_split]
            results = torch.cat(results, 0)
        else:
            results = torch.zeros(x.shape[0]).to(device)  # 修改
            res = [None] * len(x_split)

            for i, x in enumerate(x_split):
                res[i] = self.pool.submit(self._predict, x)
            for i, a in enumerate(res):
                result = a.result()
                results[i * 30: i * 30 + result.shape[0]] = result
        return results

    def _predict(self, x):
        with torch.no_grad():
            pred = self.model(x)
            # pred = torch.softmax(pred, 1)  # 暂时更改
            pred = torch.argmax(pred, 1)
            return pred

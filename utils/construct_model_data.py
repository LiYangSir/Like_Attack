from utils.generate_model import ImageModel
from utils.load_data import ImageData, split_data
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def construct_model_and_data(args):
    model = ImageModel(args.arch, args.dataset)
    loader = ImageData(args.data, args.dataset, num_samples=args.num_samples).data_loader
    target, label = next(iter(loader))
    target, label = target.to(device), label.to(device)
    x_data, y_data = split_data(target, label, model, args.num_classes)
    result = {'data_model': model, 'x_data': x_data, 'y_data': y_data, 'clip_min': 0.0, 'clip_max': 1.0}
    if args.attack_type == 'targeted':
        np.random.seed(0)
        length = y_data.shape[0]
        idx = [np.random.choice([j for j in range(length) if j != label]) for label in range(length)]
        result['target_labels'] = y_data[idx]
        result['target_images'] = x_data[idx]

    return result

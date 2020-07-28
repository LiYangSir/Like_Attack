import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from dataset.imagenet_dataset import ImageNetDataset


class ImageData:
    def __init__(self, path, dataset_name='mnist', num_samples=32, num_works=0, trans=transforms.ToTensor()):
        if dataset_name == 'mnist':
            dataset = datasets.MNIST(root=path, download=True, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif dataset_name == 'cifar10':
            dataset = datasets.CIFAR10(root=path, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        else:
            dataset = ImageNetDataset(root=path, transform=transforms.Compose([
                transforms.Resize((255, 255)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]))

        self.data_loader = DataLoader(dataset, shuffle=True, batch_size=num_samples, num_workers=num_works)


def split_data(x, y, model, num_classes=10):
    pred = model.predict(x)
    correct_idx = y == pred
    print('Accuracy is {}'.format(np.mean(np.where(correct_idx.cpu().numpy(), 1, 0))))
    label_pred = pred[correct_idx]
    x, y = x[correct_idx], label_pred

    x_data, y_data = [], []
    for class_id in range(num_classes):
        _x = x[label_pred == class_id]
        _y = y[label_pred == class_id]

        x_data.append(_x)
        y_data.append(_y)

    x_data = torch.cat(x_data, 0)
    y_data = torch.cat(y_data, 0)

    return x_data, y_data

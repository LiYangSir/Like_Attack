import argparse
import os
import csv
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import models, transforms, datasets
from utils.generate_model import ImageModel
from torch import nn
from torch import optim
import time
import pandas as pd
import scipy.misc
import torch
import numpy as np
from config.config import cifar100_classes
from matplotlib import pyplot as plt
from utils.attack_setting import load_pgen

parser = argparse.ArgumentParser(description='PyTorch Black Attack Test')
parser.add_argument('--data', metavar='DIR', default="./output", help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152')
parser.add_argument('--dataset', default='imagenet', help='use trained model')
parser.add_argument('--attack_type', type=str, choices=['targeted', 'untargeted'], default='untargeted')
parser.add_argument('--batch_size', type=int, default=10)

args = parser.parse_args()
print(args)


def generate_csv(path):
    csv_file = open("./data/imagenet.csv", "w", newline="")
    writer = csv.writer(csv_file, dialect='excel')
    for root, dirs, files in os.walk(path):
        for d in dirs:
            for root2, dirs2, files2 in os.walk(os.path.join(path, d)):
                for f in files2:
                    list = []
                    str = os.path.join(os.path.join(root, d), f)
                    list.append(str.replace('\\', '/'))
                    list.append(d)
                    writer.writerows([list])
    csv_file.close()


def print_format(title, number):
    print("| {:20} | {:<10}|".format(title, number))


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_data(path, time):
    df = pd.read_excel(path)
    for i in range(4):
        if df.loc[i * 3, 1] == 'center':
            queries = df.loc[i * 3 + 1, :].values
            distance = df.loc[i * 3 + 2, :].values
            np.save(f'npy/{time}_center_distance.npy', distance)
            np.save(f'npy/{time}_center_queries.npy', queries)
        elif df.loc[i * 3, 1] == 'DCT':
            queries = df.loc[i * 3 + 1, :].values
            distance = df.loc[i * 3 + 2, :].values
            np.save(f'npy/{time}_DCT_distance.npy', distance)
            np.save(f'npy/{time}_DCT_queries.npy', queries)
        elif df.loc[i * 3, 1] == 'resize':
            queries = df.loc[i * 3 + 1, :].values
            distance = df.loc[i * 3 + 2, :].values
            np.save(f'npy/{time}_resize_distance.npy', distance)
            np.save(f'npy/{time}_resize_queries.npy', queries)
        elif df.loc[i * 3, 1] == 'random':
            queries = df.loc[i * 3 + 1, :].values
            distance = df.loc[i * 3 + 2, :].values
            np.save(f'npy/{time}_random_distance.npy', distance)
            np.save(f'npy/{time}_random_queries.npy', queries)


if __name__ == '__main__':
    # p1 = torch.randn((5, 3, 224, 224))
    # # [scipy.misc.imsave(f'./config/original_{i}.png', img.permute(1, 2, 0).numpy()) for i, img in enumerate(p1)]
    # p_gen = load_pgen("cifar", 'resize', 'l2')
    # result = p_gen.generate_ps(p1, 5)
    # result = torch.mean(result * 0.6, 0).unsqueeze(0)
    # [scipy.misc.imsave(f'./config/{i}.png', img.permute(1, 2, 0).numpy()) for i, img in enumerate(result)]
    # time = 'second'
    # load_data(f'csv/{time}.xlsx', time)
    # pass

    # dataset = datasets.MNIST(root='./data/', download=True, train=True, transform=transforms.Compose([
    #     transforms.ToTensor()]))
    dataset = datasets.CIFAR100(root='./data/', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor()]))
    data_loader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=0)
    target, label = next(iter(data_loader))
    classes = [cifar100_classes[i.item()] for i in label]
    imshow(make_grid(target))
    model = ImageModel("resnet", "cifar100")
    target = target.cuda()
    pred = model.predict(target)
    print(pred)
    print(label)

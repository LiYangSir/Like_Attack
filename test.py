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
import numpy as np
from config.config import cifar100_classes
from matplotlib import pyplot as plt

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


if __name__ == '__main__':
    # dataset = datasets.MNIST(root='./data/', download=True, train=True, transform=transforms.Compose([
    #     transforms.ToTensor()]))
    dataset = datasets.CIFAR100(root='./data/', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor()]))
    data_loader = DataLoader(dataset, shuffle=False, batch_size=5, num_workers=0)
    target, label = next(iter(data_loader))
    classes = [cifar100_classes[i.item()] for i in label]
    imshow(make_grid(target))
    model = ImageModel("densenet", "cifar100")
    pred = model.predict(target)
    print(model)

    # print_format("original_label", 15)
    # generate_csv('./data/imagenet')
    # image_dataset = ImageFolder(root=args.data, transform=transforms.ToTensor())
    # image_loader = DataLoader(image_dataset, batch_size=args.batch_size)
    # model = ImageModel(args.arch, args.dataset)
    # for image, target in image_loader:
    #     pred = model.predict(image)
    #     print(pred)

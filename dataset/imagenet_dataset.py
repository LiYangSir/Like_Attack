import torch
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image
import csv


class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        root = root + 'imagenet.csv'
        train_data_paths = []
        train_data_labels = []
        with open(root)as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                train_data_paths.append(row[0])
                train_data_labels.append(int(row[1]))
        self.img_path_list = train_data_paths
        self.label_path_list = train_data_labels
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_path_list[idx])
        label = self.label_path_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)  # 是否进行transform

        return image, label

    def __len__(self):
        return len(self.img_path_list)

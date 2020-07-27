import torch
from torch import nn
from torch.functional import F


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1, 1)
        self.conv3 = nn.Conv2d(50, 70, 3, 1, 1)
        self.fc1 = nn.Linear(7 * 7 * 70, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 7 * 7 * 70)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

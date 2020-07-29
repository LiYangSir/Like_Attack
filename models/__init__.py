from torchvision import models, transforms, datasets

from torchvision.models.resnet import resnet152 as resnet152_imagenet
from torchvision.models.resnet import resnet101
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnext101_32x8d
from torchvision.models.vgg import vgg19_bn
from torchvision.models.densenet import densenet169 as densenet169_imagenet
from torchvision.models.inception import inception_v3
from models.mnist_vgg import vgg as vgg_mnist
from models.mnist_resnet import MnistResNet as resnet_mnist
from models.cifar10_densenet import densenet as densenet_cifar10
from models.cifar100_densenet import densenet_cifar100 as densenet_cifar100
from models.cifar10_resnet import ResNet as resnet_cifar10
from models.cifar100_resnet import ResNet as resnet_cifar100
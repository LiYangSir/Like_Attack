from models.cifar10_densenet import densenet


def densenet_cifar100():
    return densenet(3, 100, 16, [6, 14])

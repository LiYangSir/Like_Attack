import numpy as np
from matplotlib import pyplot as plt
import os
from config.config import *
import torch
import scipy.misc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_format(data):
    print("+----------------------+-----------+")
    print("| parameter            | number    |")
    print("+----------------------+-----------+")
    for i in list(data.items()):
        print("| {:20} | {:<10}|".format(i[0], i[1]))
    print("+----------------------+-----------+")
    print()


def grey_and_rgb(original_image):
    if original_image.shape[-1] == 1:
        return np.concatenate((original_image, original_image, original_image), axis=-1)
    else:
        return original_image


def show_and_save(data, dataset, distance_list, queries_list, show=False, path='./output', file_name='fig.png'):
    fig = plt.figure(figsize=(9, 8))
    a1 = fig.add_subplot(231)
    a2 = fig.add_subplot(232)
    a3 = fig.add_subplot(233)
    a4 = fig.add_subplot(2,3,(4,6))


    if dataset == 'mnist':
        classes = mnist_classes
    elif dataset == 'cifar10':
        classes = cifar10_classes
    elif dataset == 'cifar100':
        classes = cifar100_classes
    else:
        classes = imagenet_classes

    a1.imshow(np.clip(data['original_image'], 0, 1))
    a1.set_xlabel('original_label : {}'.format(classes[data['original_label']]))
    a1.set_title('original_image')
    a1.set_xticks([])
    a1.set_yticks([])

    if data['target_label'] is None:
        fig.suptitle('UnTarget {} ({} , {}) in Method CENTER'.format(data['constraint'], data['model_name'],data['dataset']))
        a2.imshow(np.clip(1.0 - (data['disturb_image'] - data['original_image']), 0, 1))
        a2.set_title('sub_image')
    else:
        fig.suptitle('Target: From {} To {} ,{} ({} , {}) in Method CENTER'.format(classes[data['original_label']], classes[data['target_label']], data['constraint'], data['model_name'],data['dataset']))
        a2.imshow(np.clip(data['target_image'], 0, 1))
        a2.set_title('target_image')
        a2.set_xlabel('target_label : {}'.format(classes[data['target_label']]))

    a2.set_xticks([])
    a2.set_yticks([])

    a3.imshow(np.clip(data['disturb_image'], 0, 1))
    a3.set_xlabel('disturb_label : {}'.format(classes[data['disturb_label']]))
    a3.set_title('disturb_image')
    a3.set_xticks([])
    a3.set_yticks([])

    a4.plot(queries_list, distance_list, 'b')
    a4.set_xlabel("queries")
    a4.set_ylabel("distance")
    a4.set_xlim([0, 1000])
    # a4.set_ylim([0, 11])

    fig.savefig(os.path.join(path, file_name))
    if show:
        plt.show()
    plt.close(fig)


def imshow(inp):
    inp = inp[0].cpu().numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.show()


def grey_rgb(original_image):
    if original_image.shape[1] == 1:
        return torch.cat((original_image, original_image, original_image), 1)
    else:
        return original_image


def save(target, disturb, dataset, network):
    if dataset == 'mnist':
        image = torch.cat([grey_rgb(target), torch.zeros(1, 3, 28, 8).to(device), grey_rgb(disturb)], 3)
    elif dataset == 'cifar10':
        image = torch.cat([target, torch.zeros(1, 3, 32, 8).to(device), disturb], 3)
    elif dataset == 'cifar100':
        image = torch.cat([target, torch.zeros(1, 3, 32, 8).to(device), disturb], 3)
    else:
        image = torch.cat([target, torch.zeros(1, 3, 224, 8).to(device), disturb], 3)
    scipy.misc.imsave('./output/{}/{}/result/{}.jpg'.format(dataset, network, i),
                      image[0].cpu().numpy().transpose((1, 2, 0)))

# def show_image(queries,distance):
#     # plt.plot(queries,distance)
#     # plt.xlabel("Number of Queries")
#     # plt.ylabel("l2 distance")
#     # plt.show()
#     return np.array(queries),np.array(distance)

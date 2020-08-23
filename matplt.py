from matplotlib import pyplot as plt
import numpy as np
import argparse
from scipy.signal import signaltools


def np_move_avg(a, n, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Pic')
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--dataset', default='cifar100', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--limited_query', type=int, default=1000)
    parser.add_argument('--query', type=int, default=0)
    parser.add_argument('--constraint', type=str, default='linf', choices=['l2', 'linf'])
    parser.add_argument('--attack_type', type=str, default='untargeted')
    parser.add_argument('--change', type=str, default='random center')
    # parser.add_argument('--gradient_strategy', type=str, default="DCT",
    #                     choices=['centerconv', 'random', 'DCT', 'upsample'])

    args = parser.parse_args()
    print(args)
    # for query in range(10):  # 1: 0  2:1  3: 9  4: 3
    # for query in [3]:  # 1: 8  2:3  3: 9  4: 2
    #     attack_type = ''
    #     if args.attack_type == 'untargeted':
    #         attack_type = '_untar'
    #     par = list()
    #     data = []
    #     par.append(f'distance{query}')
    #     par.append(f'queries{query}')
    #     for strategy in ['center', 'random', 'DCT', 'resize']:
    #         data.clear()
    #         for i in par:
    #             if args.arch == 'vgg':
    #                 path = f'npy/{args.arch}/{args.arch}_{args.dataset}_{args.limited_query}_{strategy}_{args.constraint}{attack_type}/{i}.npy'
    #             else:
    #                 path = f'npy/{args.arch}/{args.dataset}/{args.arch}_{args.dataset}_{args.limited_query}_{strategy}_{args.constraint}{attack_type}/{i}.npy'
    #             if i.startswith('distance'):
    #                 ss = signaltools.medfilt(np.load(path))
    #                 # ss = np_move_avg(np.load(path), 3, mode='valid')
    #                 data.append(ss)
    #             else:
    #                 data.append(np.load(path))
    #         np.savetxt(f'csv/4/{strategy}.csv', [data[1], data[0]], fmt='%s', delimiter=',')
    #         plt.plot(data[1], data[0], label=strategy if strategy is not 'resize' else 'upsample')
    #     plt.yscale("log")
    #
    #     plt.title(r'${} \ l_ \infty ({}, {})$'.format(args.attack_type, args.dataset, args.arch), fontdict={'size': 20})
    #     plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=2, prop={'size': 15})
    #     plt.tick_params(labelsize=20)
    #     # plt.savefig("./t31.png")
    #     plt.show()
    for i in ['second']:
        for j in ['DCT', 'random', 'resize', 'center']:
            data = []
            if i == 'first' or i == 'third':
                attack_type = 'Targeted'
            else:
                attack_type = 'UnTargeted'
            if i == 'first' or i == 'second':
                sa = 'ResNet, Cifar10'
            else:
                sa = 'ResNet, Cifar100'
            distance = np.load(f'npy/{i}_{j}_distance.npy', allow_pickle=True)
            queries = np.load(f'npy/{i}_{j}_queries.npy', allow_pickle=True)
            if j == 'DCT':
                plt.plot(queries, distance, 'cyan', label=j.upper())
            elif j == 'resize':
                plt.plot(queries, distance, label="upsample".upper())
            else:
                plt.plot(queries, distance, label=j.upper())
        plt.yscale("log")
        plt.title(f'${attack_type} \ l_ \infty ({sa})$', fontdict={'size': 20})
        plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=2, prop={'size': 10})
        plt.tick_params(labelsize=20)
        plt.show()

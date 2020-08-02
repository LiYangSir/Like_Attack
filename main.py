from utils.generate_model import ImageModel
from utils.load_data import ImageData, split_data
import argparse
import numpy as np
import scipy.misc
import models
from like_attack import LikeAttack
from utils.generate_video import video
from utils.attack_setting import *
from utils.show_or_save import *
from utils import construct_model_and_data

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Black Attack')
parser.add_argument('--data', metavar='DIR', default="./data/", help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--dataset', default='mnist', help='please choice dataset',
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--limited_query', type=int, default=1000, help='limited quety time')
parser.add_argument('--constraint', type=str, choices=['l2', 'linf'], default='l2')
parser.add_argument('--attack_type', type=str, choices=['targeted', 'untargeted'], default='untargeted')
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--show', default=False, action="store_true")
parser.add_argument('--threadPool', default=False, action="store_true")
parser.add_argument('--atk_level', type=int, default=999)
parser.add_argument('--gradient_strategy', type=str, default="DCT", choices=['resize', 'random', 'DCT'])
parser.add_argument('--stepsize_search', type=str, choices=['geometric_progression', 'grid_search'],
                    default='geometric_progression')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def construct_model_and_data():
    model = ImageModel(args.arch, args.dataset)
    loader = ImageData(args.data, args.dataset, num_samples=args.num_samples).data_loader
    target, label = next(iter(loader))
    target, label = target.to(device), label.to(device)
    x_data, y_data = split_data(target, label, model, args.num_classes)
    result = {'data_model': model,
              'x_data': x_data,
              'y_data': y_data,
              'clip_min': 0.0,
              'clip_max': 1.0}
    if args.attack_type == 'targeted':
        np.random.seed(0)
        length = y_data.shape[0]
        idx = [np.random.choice([j for j in range(length) if j != label]) for label in range(length)]
        result['target_labels'] = y_data[idx]
        result['target_images'] = x_data[idx]

    return result


def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.show()


def grey_and_rgb(original_image):
    if original_image.shape[1] == 1:
        return torch.cat((original_image, original_image, original_image), 1)
    else:
        return original_image


def save(target, disturb, dataset, network):
    if dataset == 'mnist':
        image = torch.cat([grey_and_rgb(target), torch.zeros(1, 3, 28, 8).to(device), grey_and_rgb(disturb)], 3)
    elif dataset == 'cifar10':
        image = torch.cat([target, torch.zeros(1, 3, 32, 8).to(device), disturb], 3)
    elif dataset == 'cifar100':
        image = torch.cat([target, torch.zeros(1, 3, 32, 8).to(device), disturb], 3)
    else:
        image = torch.cat([target, torch.zeros(1, 3, 224, 8).to(device), disturb], 3)
    scipy.misc.imsave('./output/{}/{}/result/{}.jpg'.format(dataset, network, i),
                      image[0].cpu().numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    result = construct_model_and_data(args)
    model = result['data_model']

    for i, target in enumerate(result['x_data']):

        target = torch.unsqueeze(target, 0)
        if args.attack_type == 'targeted':
            target_label = result['target_labels'][i]
            target_image = result['target_images'][i]
            target_image = torch.unsqueeze(target_image, 0)
        else:
            target_label = None
            target_image = None
        p_gen = load_pgen(args.dataset, args.gradient_strategy, args.constraint)
        like_attack = LikeAttack(model, target, iter=i, limited_query=args.limited_query, clip_max=1, clip_min=0,
                                 constraint=args.constraint, dataset=args.dataset, mask=None,
                                 rv_generator=p_gen, gamma=1.0, target_label=target_label, target_image=target_image,
                                 stepsize_search=args.stepsize_search, max_num_evals=1e4, init_num_evals=10,
                                 show_flag=args.show, atk_level=args.atk_level)
        disturb_image = like_attack.attack()
        # save(target, disturb_image, args.dataset, args.arch)
        print("generate_video...")
        video(args.dataset, args.arch, i)

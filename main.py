import argparse
import models
from like_attack import LikeAttack
from utils.generate_video import video
from utils.attack_setting import *
from utils.show_or_save import *
from utils import construct_model_and_data
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(models.__dict__[name]))

"""
注意修改以下几点
    1. 数据集
    2. 网络模型
    3. 数据集
    4. 输出类别
    6. 梯度方式
    5. 目标攻击还是无目标攻击
"""
parser = argparse.ArgumentParser(description='PyTorch Black Attack')

parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet169', choices=model_names)
parser.add_argument('--dataset', default='imagenet', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--limited_query', type=int, default=1000)
parser.add_argument('--constraint', type=str, choices=['l2', 'linf'], default='l2')
parser.add_argument('--attack_type', type=str, choices=['targeted', 'untargeted'], default='untargeted')
parser.add_argument('--gradient_strategy', type=str, default="DCT", choices=['resize', 'random', 'DCT'])

parser.add_argument('--data', metavar='DIR', default="./data/", help='path to dataset')
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--show', default=False, action="store_true")
parser.add_argument('--threadPool', default=False, action="store_true")
parser.add_argument('--atk_level', type=int, default=999)
parser.add_argument('--stepsize_search', type=str, choices=['geometric_progression', 'grid_search'],
                    default='geometric_progression')
args = parser.parse_args()
print(args)

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
        # print("generate_video...")
        # video(args.dataset, args.arch, i)

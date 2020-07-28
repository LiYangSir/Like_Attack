from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import scipy.misc
import matplotlib.pyplot as plt
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_distance(original_image, disturb_image, function='l2'):
    if function == 'l2':
        return torch.norm(original_image - disturb_image)
    else:
        return torch.max(abs(original_image - disturb_image))


def print_format(data):
    print("+----------------------+-----------+")
    print("| parameter            | number    |")
    print("+----------------------+-----------+")
    for i in list(data.items()):
        print("| {:20} | {:<10}|".format(i[0], i[1]))
    print("+----------------------+-----------+")
    print()


def clamp_image(image, min, max):
    return torch.clamp(image, min, max)


def grey_and_rgb(original_image):
    if original_image.shape[-1] == 1:
        return np.concatenate((original_image, original_image, original_image), axis=-1)
    else:
        return original_image


def show_and_save(data, show=False, path='./output', file_name='fig.png'):
    fig = plt.figure(figsize=(9, 4))
    a1 = fig.add_subplot(131)
    a2 = fig.add_subplot(132)
    a3 = fig.add_subplot(133)

    a1.imshow(np.clip(data['original_image'], 0, 1))
    a1.set_xlabel('original_label : {}'.format(data['original_label']))
    a1.set_title('original_image')
    a1.set_xticks([])
    a1.set_yticks([])

    if data['target_label'] is None:
        fig.suptitle('UnTarget')
        a2.imshow(np.clip(data['disturb_image'] - data['original_image'], 0, 1))
        a2.set_title('sub_image')
    else:
        fig.suptitle('Target: {} -> {}'.format(data['original_label'], data['target_label']))
        a2.imshow(np.clip(data['target_image'], 0, 1))
        a2.set_title('target_image')

    a2.set_xticks([])
    a2.set_yticks([])

    a3.imshow(np.clip(data['disturb_image'], 0, 1))
    a3.set_xlabel('disturb_label : {}'.format(data['disturb_label']))
    a3.set_title('disturb_image')
    a3.set_xticks([])
    a3.set_yticks([])

    fig.savefig(os.path.join(path, file_name))
    if show:
        plt.show()
    plt.close(fig)


class Like_Attack:

    def __init__(self, model, original_image, iter, limited_query=1000, clip_max=1, clip_min=0, constraint='l2',
                 num_iterations=40, dataset='mnist',
                 gamma=1.0, target_label=None, target_image=None, stepsize_search='geometric_progression',
                 max_num_evals=1e4, init_num_evals=10, verbose=True, show_flag=False):
        self.model = model
        self.original_image = original_image
        self.iter = iter
        self.dataset = dataset
        self.limited_query = limited_query
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.constraint = constraint
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.target_label = target_label
        self.target_image = target_image
        self.step_size_search = stepsize_search
        self.max_num_eval = max_num_evals
        self.init_num_eval = init_num_evals
        self.verbose = verbose
        self.original_label = model.predict(original_image)
        self.d = int(np.prod(original_image.shape))
        self.shape = original_image.shape
        self.cur_iter = 0
        self.queries = 0
        self.show_flag = show_flag
        if constraint == 'l2':
            self.theta = gamma / (np.sqrt(self.d) * self.d)
        elif constraint == 'linf':
            self.theta = gamma / (self.d ** 2)

    def attack(self):
        if not os.path.exists('./output/{}/{}/{}'.format(self.dataset, self.model.model_name, self.iter)):
            os.makedirs('./output/{}/{}/{}'.format(self.dataset, self.model.model_name, self.iter))
        # if not os.path.exists('./output/{}/result/'.format(self.dataset)):
        #     os.makedirs('./output/{}/{}/result/'.format(self.dataset, self.model.model_name))
        if self.verbose:
            print("original_label : ", int(self.original_label.item()))
            if self.target_label is not None:
                print("target label : ", int(self.target_label.item()))
            # scipy.misc.imsave('./output/{}/{}/original.jpg'.format(self.dataset, self.iter),
            #                   grey_and_rgb(self.original_image[0].cpu().permute(1, 2, 0).numpy()))
        disturb_image = self.initialize()
        disturb_image, distance = self.binary_search_batch(disturb_image)
        dist = compute_distance(disturb_image, self.original_image, self.constraint)
        i = -1
        # for i in range(self.num_iterations):
        while self.queries <= self.limited_query:
            i += 1
            start_time = time.time()
            self.cur_iter = i + 1
            delta = self.select_delta(distance)

            num_eval = int(self.init_num_eval * np.sqrt(i + 1))
            num_eval = int(min([num_eval, self.max_num_eval]))

            grad = self.approximate_gradient(disturb_image, num_eval, delta)  # 可能限制溢出
            if self.queries > self.limited_query:
                break

            if self.constraint == 'linf':
                update = torch.sign(grad)
            else:
                update = grad
            if self.step_size_search == 'geometric_progression':

                epsilon = self.geometric_progression_for_stepsize(disturb_image, update, dist)  # 可能限制溢出
                if self.queries > self.limited_query:
                    break

                disturb_image = clamp_image(disturb_image + epsilon * update, self.clip_min, self.clip_max)
                disturb_image, distance = self.binary_search_batch(disturb_image)  # 可能限制溢出
                if self.queries > self.limited_query:
                    break

            elif self.step_size_search == 'grid_search':
                pass

            dist = compute_distance(disturb_image, self.original_image, self.constraint)
            end_time = time.time()
            if self.verbose:
                disturb_label = self.model.predict(disturb_image)
                print_data = {
                    'iteration': i,
                    'distance': round(dist.item(), 3),
                    'disturb_label': int(disturb_label.item()),
                    'queries': self.queries,
                    'spend time': round((end_time - start_time), 3)
                }
                print_format(print_data)
                # scipy.misc.imsave('./output/{}/{}/disturb_{}.jpg'.format(self.dataset, self.iter, i),
                # grey_and_rgb(disturb_image[0].cpu().permute(1, 2, 0).numpy()))
                data = {
                    'disturb_image': grey_and_rgb(disturb_image[0].cpu().permute(1, 2, 0).numpy()),
                    'disturb_label': int(disturb_label.item()),
                    'target_image': grey_and_rgb(self.target_image[0].cpu().permute(1, 2, 0).numpy()),
                    'target_label': int(self.target_label.item()),
                    'original_image': grey_and_rgb(self.original_image[0].cpu().permute(1, 2, 0).numpy()),
                    'original_label': int(self.original_label.item()),
                }
                show_and_save(data, show=self.show_flag,
                              path='./output/{}/{}/{}'.format(self.dataset, self.model.model_name, self.iter),
                              file_name='result_{}.png'.format(i))

        return disturb_image

    def geometric_progression_for_stepsize(self, x, update, dist):
        epsilon = dist / np.sqrt(self.cur_iter)

        def phi(epsilon):
            new = x + epsilon * update
            return self.decision_function(new)

        while self.queries <= self.limited_query and (not phi(epsilon)):
            epsilon /= 2.0
        return epsilon

    def approximate_gradient(self, sample, num_eval, delta):
        noise_shape = [num_eval] + list(self.shape[1:])

        if self.constraint == 'l2':
            rv = torch.randn(*noise_shape).to(device)
        elif self.constraint == 'linf':
            rv = torch.rand(noise_shape).to(device)

        rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=(1, 2, 3), keepdim=True))
        disturb_image = sample + delta * rv
        disturb_image = clamp_image(disturb_image, self.clip_min, self.clip_max)
        rv = (disturb_image - sample) / delta

        decision = self.decision_function(disturb_image)
        decision_shape = [len(decision)] + [1] * (len(self.shape) - 1)
        decision = torch.where(decision, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
        f_val = 2 * decision.reshape(decision_shape) - 1.0

        if torch.mean(f_val) == 1.0:
            grad = torch.mean(rv, dim=0)
        elif torch.mean(f_val) == -1.0:
            grad = -torch.mean(rv, dim=0)
        else:
            f_val -= torch.mean(f_val)
            grad = torch.mean(f_val * rv, dim=0)
        grad = grad / torch.norm(grad)

        return grad

    def initialize(self):
        num_evals = 0
        if self.target_image is None:
            while True:
                random_noise = torch.rand(self.shape).to(device)
                success = self.decision_function(random_noise)
                num_evals += 1
                if success:
                    break
                assert num_evals < 1e4, "Initialization failed!"

            # 二分查找
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * self.original_image + mid * random_noise

                success = self.decision_function(blended)
                if success:
                    high = mid
                else:
                    low = mid
            initialization = (1 - high) * self.original_image + high * random_noise
        else:
            initialization = self.target_image
        return initialization

    def binary_search_batch(self, disturb_image):
        distance = compute_distance(self.original_image, disturb_image, self.constraint)
        if self.constraint == 'linf':
            highs = distance
            threshold = np.minimum(distance * self.theta, self.theta)
        else:
            highs = 1.0
            threshold = self.theta

        lows = 0.0

        while (highs - lows) / threshold > 1:
            mid = (highs + lows) / 2
            mid_images = self.project(disturb_image, mid)

            decision = self.decision_function(mid_images)
            if self.queries > self.limited_query:
                return disturb_image, distance  # 随意返回
            lows = np.where(~decision.cpu().numpy(), mid, lows)[0]
            highs = np.where(decision.cpu().numpy(), mid, highs)[0]

        out_image = self.project(disturb_image, highs)

        dist = compute_distance(self.original_image, out_image, self.constraint)

        dist = distance  # 这里应该是dist
        out_image = out_image
        return out_image, dist

    def project(self, disturb_image, alphas):
        if self.constraint == 'l2':
            return (1 - alphas) * self.original_image + alphas * disturb_image
        elif self.constraint == 'linf':
            return clamp_image(disturb_image, self.original_image - alphas, self.original_image + alphas)

    def decision_function(self, image):
        self.queries += 1
        images = clamp_image(image, self.clip_min, self.clip_max)
        pred = self.model.predict(images)

        if self.target_label is None:
            return pred != self.original_label
        else:
            return pred == self.target_label

    def select_delta(self, distance):
        if self.cur_iter == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.constraint == 'l2':
                delta = np.sqrt(self.d) * self.theta * distance
            else:
                delta = self.d * self.theta * distance
        return delta

from utils.gradient_strategy.strategy import Strategy
import torch
from scipy.signal import butter, filtfilt
import numpy as np
import cv2
from scipy import fftpack
from torch import nn
from torchvision import transforms
from torch.functional import F
from sklearn.decomposition import PCA
import cv2


class KLGenerator(Strategy):
    def __init__(self, factor, *shape):
        super(KLGenerator, self).__init__()
        self.factor = factor

    def generate_ps(self, inp, num_eval, level=0.1):
        inp = inp[0].cpu().numpy()
        channel, height, width = inp.shape
        ps = []
        for _ in range(num_eval):
            p_signal = np.zeros((height, width, channel))
            for c in range(channel):
                rv = np.random.randn(height, width, 1)
                data_mat = np.mat(rv)
                low_data_mat, recon_mat = self.im_PCA(data_mat, 0.75)
                p_signal[:, :, c] = recon_mat
            ps.append(p_signal)
        ps = np.stack(ps, 0)
        ps = torch.from_numpy(ps).permute(0, 3, 1, 2).type(torch.float32).to(self.device)
        return ps

    def eigValPct(self, eigVals, percentage):
        sortArray = np.sort(eigVals)[::-1]  # 特征值从大到小排序
        pct = np.sum(sortArray) * percentage
        tmp = 0
        num = 0
        for eigVal in sortArray:
            tmp += eigVal
            num += 1
            if tmp >= pct:
                return num

    def im_PCA(self, data_mat, percentage=0.9):
        mean_val = np.mean(data_mat, axis=0)
        mean_removed = data_mat - mean_val
        cov_mat = np.dot(np.transpose(mean_removed), mean_removed)

        eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
        k = self.eigValPct(eig_vals, percentage)  # 要达到方差的百分比percentage，需要前k个向量

        eig_val_ind = np.argsort(eig_vals)[::-1]
        eig_val_ind = eig_val_ind[:k]
        red_eig_vects = eig_vects[:, eig_val_ind]
        low_d_data_mat = mean_removed * red_eig_vects
        recon_mat = (low_d_data_mat * red_eig_vects.T) + mean_val
        return low_d_data_mat, recon_mat

    def im_pca(self, data_mat, percentage=0.9):
        mean_val = np.mean(data_mat, axis=0)
        mean_removed = data_mat - mean_val
        cov_mat = np.dot(np.transpose(mean_removed), mean_removed)

        eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
        eig_signal = np.zeros(eig_vects.shape[0], eig_vects.shape[1])
        eig_signal[:(eig_vects.shape[0]//percentage), :(eig_vects.shape[0]//percentage)] = eig_vects

        low_d_data_mat = mean_removed * eig_signal
        recon_mat = (low_d_data_mat * eig_signal.T) + mean_val
        return low_d_data_mat, recon_mat


if __name__ == '__main__':
    generator = KLGenerator(2)
    img = cv2.imread('ava.jpg')
    blue = img[:, :, 0]

    # pca = PCA(n_components=112).fit(blue)
    # # 降维
    # x_new = pca.transform(blue)
    # # 还原降维后的数据到原空间
    # recdata = pca.inverse_transform(x_new)

    dataMat = np.mat(blue)
    lowDDataMat, reconMat = generator.im_PCA(dataMat, 0.5)
    print('原始数据', blue.shape, '降维数据', lowDDataMat.shape)

    cv2.imshow('blue', blue)
    cv2.imshow('reconMat', np.array(reconMat, dtype='uint8'))
    cv2.waitKey(0)

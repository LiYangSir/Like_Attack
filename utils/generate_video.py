import os
import cv2
import numpy as np
from tqdm import tqdm
import re


def video(dataset, network, iter, fps=2, size=(900, 800)):
    if not os.path.exists('./output/video/{}/{}/'.format(dataset, network)):
        os.makedirs("./output/video/{}/{}/".format(dataset, network))

    save_path = './output/video/{}/{}/'.format(dataset, network)
    video_writer = cv2.VideoWriter(save_path + 'iter_{}.mp4'.format(iter),
                                   cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    path = './output/{}/{}/{}/'.format(dataset, network, iter)
    files = os.listdir(path)
    com = re.compile(r'\d+')
    files.sort(key=lambda x: int(com.search(x).group()))
    files = tqdm(files, desc="Generate Video")
    for i in files:
        files.set_description(f"Now Get {i}")
        img = cv2.imread(path + i)
        video_writer.write(img)

    video_writer.release()

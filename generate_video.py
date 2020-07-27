import os
import cv2
import numpy as np


def video(dataset='mnist', fps=2, size=(900, 400)):
    videoWriter = cv2.VideoWriter('./output/video/{}.mp4'.format(dataset),
                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    path = './output/{}'.format(dataset)
    for i in os.listdir(path):
        for j in range(30):
            img = cv2.imread("{}/{}/result_{}.png".format(path, i, j))
            videoWriter.write(img)
        break
    videoWriter.release()

video()

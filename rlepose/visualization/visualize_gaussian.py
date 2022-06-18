# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------
import random

import pandas as pd
import os
import pprint
import argparse
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
from torch.autograd import Variable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
import numpy as np


def generate_color_list(color_num=30):
    color_list = []
    for i in range(0, color_num):
        color_list.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    return color_list


def main():
    landmarks_frame = pd.read_csv(
        '/mnt/sd2/Keypoint_detection/HRNet-Facial-Landmark-Detection/data/scoliosis/train_ori.csv')
    title = landmarks_frame.columns[1:]

    mean = np.array([0.293, 0.293, 0.293], dtype=np.float32)
    std = np.array([0.321, 0.321, 0.321], dtype=np.float32)

    im_black = np.zeros((1024, 256, 3), dtype='uint8')
    im_black2 = np.zeros((1024, 256, 3), dtype='uint8')
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, 100)
    y = np.random.normal(mu, sigma, 100)
    color_list = generate_color_list(100)
    for j in range(len(s)):
        print(s)

        cv2.circle(im_black, (int(s[j] * 255 + 128), int(y[j] * 255 + 512)),
                   radius=2, color=color_list[j], thickness=2)
    cv2.imwrite(
        '/mnt/sd2/Keypoint_detection/HRNet-Facial-Landmark-Detection/test_result/distribution_GS.png',
        im_black)

    # = ALLsuffix
    # args.folder

    #
    # predictions = function.inference_scoliosis(config, test_loader, model)

    # torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()

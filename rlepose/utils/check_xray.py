from scipy.io import loadmat
import os
import pandas as pd
import cv2


DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
IMG_PREFIX = 'data/test'
ANN = 'labels/test'

img_dir = os.path.join(DATASET_PATH, IMG_PREFIX)

gt_path = os.path.join(DATASET_PATH, ANN)
gt_img_ann = loadmat(os.path.join(gt_path, 'sunhl-1th-01-Mar-2017-311 C AP.jpg'))['p2']
landmark_file = pd.read_csv(os.path.join(gt_path, 'landmarks.csv'), header=None)
name_file =pd.read_csv(os.path.join(gt_path, 'filenames.csv'), header=None)
# loadmat(os.path.join(gt_path, 'sunhl-1th-01-Mar-2017-311 C AP.jpg'))
print(name_file.iloc[3][0])
img = cv2.imread(os.path.join(img_dir, name_file.iloc[3][0]), cv2.IMREAD_COLOR)
h,w,c = img.shape
sp_landmark = landmark_file[3]

print(gt_img_ann)
import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box, get_center_scale
import pandas as pd

def find_pd_index(df, item):
    len = df.shape[0]
    for i in range(0, len):
        if df.iloc[i][0] == item:
            return i
    return 0

def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k + 4, :]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:, 1])
        y_inds_r = np.argsort(pt_r[:, 1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


if __name__ == "__main__":
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/offline_work_place/default1'
    kpt_json = os.path.join(exp_path, 'test_gt_kpt.json')
    # dataset_path = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/data/test'
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/test'
    ANN = 'labels/test'

    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    save_path = os.path.join(exp_path, 'val_visual')
    landmark_file = pd.read_csv(os.path.join(gt_path, 'landmarks.csv'), header=None)
    name_file = pd.read_csv(os.path.join(gt_path, 'filenames.csv'), header=None)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    kpt_w, kpt_h = (512, 1024)
    kpt_num = 68
    for i in range(len(kpt_data)):
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        # gt_img_ann = loadmat(os.path.join(gt_path, img_name))['p2']
        # gt_kpt = rearrange_pts(gt_img_ann)
        # read img
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)
        img_h, img_w = img_size[:2]
        index = find_pd_index(name_file, item=img_name)
        gt_landmark = landmark_file.iloc[index]
        gt_landmark = gt_landmark.to_numpy().reshape(2, kpt_num)
        gt_landmark = gt_landmark * [[img_w], [img_h]]
        gt_landmark = np.transpose(gt_landmark, [1,0])
        gt_kpt = rearrange_pts(gt_landmark)
        _aspect_ratio = kpt_w / kpt_h
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)
        for j in range(kpt_num):
            coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
                                         [kpt_w, kpt_h])
            img = cv2.circle(img, (int(gt_kpt[j][0]),
                                   int(gt_kpt[j][1])),
                             radius=3, color=[0, 255, 0], thickness=3)
            img = cv2.circle(img, (int(coord_draw[0]), int(coord_draw[1])),
                             radius=3, color=[0, 0, 255], thickness=3)

        cv2.imwrite(os.path.join(save_path, img_name), img)

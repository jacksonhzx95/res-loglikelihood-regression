import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds
from rlepose.utils import cobb_evaluate
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box, get_center_scale
import pandas as pd
from rlepose.utils.landmark_statistics import LandmarkStatistics


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
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/exp/ce_regress18_FB_lr_ftAug11-512x512_res18_ce_regress-flowB.yaml'
    # exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/offline_work_place/default'
    kpt_json = os.path.join(exp_path, 'test_gt_kpt.json')
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/face_landmark/'
    IMG_PREFIX = 'RawImage/ALL'
    ANN = '400_senior'
    ANN2 = '400_junior'

    gt_path = os.path.join(DATASET_PATH, ANN)
    gt_path2 = os.path.join(DATASET_PATH, ANN2)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    # kpt_data = kpt_json
    kpt_h, kpt_w = 512, 512  # (512, 256) or (1024, 512)
    kpt_num = 19
    pr_cobb_angles = []
    gt_cobb_angles = []
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()
    testing_dataset = os.path.join(DATASET_PATH, 'RawImage/Test2Data')
    TestImage_list = os.listdir(testing_dataset)
    # print(TestImage_list)
    # print(len(kpt_data))
    for i in range(len(kpt_data)):

        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        if img_name not in TestImage_list:
            # print(img_name)
            continue
        # load gt ann
        annoFolder = os.path.join(gt_path, img_name[:-4] + '.txt')
        annoFolder2 = os.path.join(gt_path2, img_name[:-4] + '.txt')
        pts1 = []
        pts2 = []
        with open(annoFolder, 'r') as f:
            lines = f.readlines()
            for i in range(kpt_num):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts1.append(coordinates_int)
        with open(annoFolder2, 'r') as f:
            lines = f.readlines()
            for i in range(kpt_num):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts2.append(coordinates_int)
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        gt_kpt = (pts1 + pts2) / 2

        # gt_kpt = np.array(pts)

        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)

        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)

        # center_beta = np.array([kpt_w_beta * 0.5, kpt_h_beta * 0.5])
        for j in range(kpt_num):
            coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
                                         [kpt_w, kpt_h])
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpt[j][0]), int(gt_kpt[j][1])))
            # landmark_dist.append(abs(coord_draw[0] - gt_kpt[j][0] + (coord_draw[1] - gt_kpt[j][1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpt[j][0]) ** 2 + (coord_draw[1] - gt_kpt[j][1]) ** 2))
        landmark_statistic.add_landmarks(image_id=img_name, predicted=pred_pts, groundtruth=gt_pts, spacing=0.1)
    overview_string = landmark_statistic.get_overview_string([2.0, 3.0, 4.0])
    print(overview_string)

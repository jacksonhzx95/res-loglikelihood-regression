import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds
from rlepose.utils import cobb_evaluate
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


def SMAPE_single_angle(gt_cobb_angles, pr_cobb_angles):
    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = out_abs
    term2 = out_add

    term2[term2 == 0] += 1e-5

    SMAPE = np.mean(term1 / term2 * 100)
    return SMAPE


if __name__ == "__main__":
    # exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/exp/default-1024x512_res50_scoliosic_regress-flow.yaml'
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/offline_work_place/default'
    # exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/exp/wo_nf_beta1-1024x512_res50_scoliosic_regress.yaml'
    kpt_json = os.path.join(exp_path, 'test_gt_kpt.json')
    # dataset_path = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/data/test'
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/test'
    ANN = 'labels/test'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    save_path = os.path.join(exp_path, 'val_visual')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    kpt_w, kpt_h = (256, 512)
    kpt_num = 68
    pr_cobb_angles = []
    gt_cobb_angles = []
    landmark_dist = []
    for i in range(len(kpt_data)):
        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        gt_img_ann = loadmat(os.path.join(gt_path, img_name))['p2']
        gt_kpt = rearrange_pts(gt_img_ann)
        # read img
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)
        img_h, img_w = img_size[:2]
        kpt_w_beta = kpt_w
        kpt_h_beta = kpt_h
        _aspect_ratio = kpt_w / kpt_h
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)
        for j in range(kpt_num):
            coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
                                         [kpt_w, kpt_h])
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpt[j][0]), int(gt_kpt[j][1])))
            # landmark_dist.append(abs(coord_draw[0] - gt_kpt[j][0] + (coord_draw[1] - gt_kpt[j][1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpt[j][0]) ** 2 + (coord_draw[1] - gt_kpt[j][1]) ** 2))

        pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pred_pts, img))

        gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_pts, img))


    pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
    gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)
    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = np.sum(out_abs, axis=1)
    term2 = np.sum(out_add, axis=1)

    SMAPE = np.mean(term1 / term2 * 100)

    print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))
    print('SMAPE is {}'.format(SMAPE))
    print('SMAPE1 is {}'.format(SMAPE_single_angle(gt_cobb_angles[:, 0], pr_cobb_angles[:, 0])))
    print('SMAPE2 is {}'.format(SMAPE_single_angle(gt_cobb_angles[:, 1], pr_cobb_angles[:, 1])))
    print('SMAPE3 is {}'.format(SMAPE_single_angle(gt_cobb_angles[:, 2], pr_cobb_angles[:, 2])))



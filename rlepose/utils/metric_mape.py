import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds
from rlepose.utils import cobb_evaluate
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box, get_center_scale
from rlepose.utils.landmark_statistics import LandmarkStatistics

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

def cal_deo(kpt_json, img_size):
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/face_landmark/'
    IMG_PREFIX = 'RawImage/TestAll'
    ANN = '400_senior'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    kpt_num = 19
    pr_cobb_angles = []
    gt_cobb_angles = []
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()
    for i in range(len(kpt_data)):
        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        # load gt ann
        annoFolder = os.path.join(gt_path, img_name[:-4] + '.txt')
        # gt_img_ann = loadmat(os.path.join(gt_path, img_name))['p2']
        # gt_kpt = rearrange_pts(gt_img_ann)
        pts = []
        with open(annoFolder, 'r') as f:
            lines = f.readlines()
            for i in range(kpt_num):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts.append(coordinates_int)
        gt_kpt = np.array(pts)
        # read img

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

    return overview_string

# def cal_mape(kpt_json, img_size)
def cal_mape(kpt_json, img_size):
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/test'
    ANN = 'labels/test'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    # kpt_file = open(kpt_json)
    # kpt_json = json.load(kpt_file)
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
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
        pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pred_pts, img))
        gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_pts, img))

    pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
    gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)
    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = np.sum(out_abs, axis=1)
    term2 = np.sum(out_add, axis=1)
    mse = np.mean(landmark_dist)
    SMAPE = np.mean(term1 / term2 * 100)
    return mse, SMAPE


def cal_mape_original(kpt_json, img_size):
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/test'
    ANN = 'labels/test'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    kpt_h, kpt_w = img_size
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
        # print(img_size)
        # modify the exact w, h ratio
        if kpt_w > img_w / img_h * kpt_h:
            kpt_w_beta = img_w / img_h * kpt_h
        elif kpt_h > img_h / img_w * kpt_w:
            kpt_h_beta = img_h / img_w * kpt_w
        scale = np.array([img_w, img_h]) * 1.25
        center = np.array([img_w * 0.5, img_h * 0.5])
        scale_beta = np.array([kpt_w_beta, kpt_h_beta])
        center_beta = np.array([kpt_w_beta * 0.5, kpt_h_beta * 0.5])
        for j in range(kpt_num):
            coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
                                         [kpt_w, kpt_h])
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpt[j][0]), int(gt_kpt[j][1])))
            landmark_dist.append(abs(coord_draw[0] - gt_kpt[j][0] + (coord_draw[1] - gt_kpt[j][1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpt[j][0]) ** 2 + (coord_draw[1] - gt_kpt[j][1]) ** 2))
        pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pred_pts, img))
        gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_pts, img))

    pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
    gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)
    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = np.sum(out_abs, axis=1)
    term2 = np.sum(out_add, axis=1)
    mse = np.mean(landmark_dist)
    SMAPE = np.mean(term1 / term2 * 100)
    return mse, SMAPE

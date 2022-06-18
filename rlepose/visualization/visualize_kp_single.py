import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds, get_affine_transform
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box, get_center_scale
import random


def generate_color_list(color_num=30):
    color_list = []
    for i in range(0, color_num):
        color_list.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    return color_list


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
    # exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/exp/default-1024x512_res50_scoliosic_regress-flow.yaml'
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/offline_work_place/default2'
    kpt_json = os.path.join(exp_path, 'test_gt_kpt.json')
    # dataset_path = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/data/test'
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/test'
    ANN = 'labels/test'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    save_path = os.path.join(exp_path, 'val_visual_off')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    kpt_w, kpt_h = (512, 1024)
    kpt_num = 68
    img_name = 'sunhl-1th-01-Mar-2017-313 B AP.jpg'
    mean_img_name = 'sunhl-1th-01-Mar-2017-312 E AP.jpg'
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, kpt_num)
    y = np.random.normal(mu, sigma, kpt_num)
    color_list = generate_color_list(kpt_num)
    gt_img_ann = loadmat(os.path.join(gt_path, img_name))['p2']
    mean_img_ann = loadmat(os.path.join(gt_path, mean_img_name))['p2']
    gt_kpt = rearrange_pts(gt_img_ann)
    mean_kpt = rearrange_pts(mean_img_ann)
    # read img
    im_black_pred = np.zeros((1024, 512, 3), dtype='uint8')
    im_black_offset = np.zeros((1024, 512, 3), dtype='uint8')
    im_black_gaussian = np.zeros((1024, 512, 3), dtype='uint8')
    im_black_mean = np.zeros((1024, 512, 3), dtype='uint8')
    im_black_offset_2 = np.zeros((1024, 512, 3), dtype='uint8')
    img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
    img_size = img.shape  # (H, W, C)
    img_h, img_w = img_size[:2]
    img_mean = cv2.imread(os.path.join(data_path, mean_img_name), cv2.IMREAD_COLOR)
    mean_img_size = img_mean.shape  # (H, W, C)
    img_mean_h, img_mean_w = img_size[:2]
    _aspect_ratio = kpt_w / kpt_h
    center, scale = get_center_scale(kpt_w, kpt_h, aspect_ratio=_aspect_ratio, scale_mult=0.8)
    mean_center, mean_scale = get_center_scale(kpt_w, kpt_h, aspect_ratio=_aspect_ratio, scale_mult=0.8)

    for j in range(kpt_num):
        coord_draw = transform_preds(np.array([gt_kpt[j][0], gt_kpt[j][1]]), center, scale,
                                     [img_w, img_h])
        coord_mean = transform_preds(np.array([mean_kpt[j][0], mean_kpt[j][1]]), mean_center, mean_scale,
                                     [img_mean_w, img_mean_h])
        im_black_pred = cv2.circle(im_black_pred, (int(coord_draw[0]),
                                                   int(coord_draw[1])),
                                   radius=3, color=color_list[j], thickness=3)
        im_black_mean = cv2.circle(im_black_mean, (int(coord_mean[0]),
                                                   int(coord_mean[1])),
                                   radius=3, color=color_list[j], thickness=3)
        im_black_gaussian = cv2.circle(im_black_gaussian, (int(s[j] * 255 + 256), int(y[j] * 255 + 512)),
                                       radius=3, color=color_list[j], thickness=3)
        im_black_offset = cv2.arrowedLine(im_black_offset, (int(coord_mean[0]), int(coord_mean[1])),
                                          (int(coord_draw[0]), int(coord_draw[1])),
                                          color=color_list[j], thickness=2)
        im_black_offset_2 = cv2.arrowedLine(im_black_offset_2, (int(coord_mean[0]), int(coord_mean[1])),
                                          (int(coord_draw[0]), int(coord_draw[1])),
                                          color=color_list[j], thickness=2)
        im_black_offset_2 = cv2.circle(im_black_offset_2, (int(coord_mean[0]),
                                                   int(coord_mean[1])),
                                   radius=2, color=color_list[j], thickness=2)
    trans_center, trans_scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.08)
    trans = get_affine_transform(trans_center, trans_scale, 0, [kpt_w, kpt_h])
    img_transed = cv2.warpAffine(img, trans, (int(kpt_w), int(kpt_h)))
    cv2.imwrite(os.path.join(save_path, 'img_transed.png'), img_transed)
    cv2.imwrite(os.path.join(save_path, 'im_black_pred.png'), im_black_pred)
    cv2.imwrite(os.path.join(save_path, 'im_black_mean.png'), im_black_mean)
    cv2.imwrite(os.path.join(save_path, 'im_black_gaussian.png'), im_black_gaussian)
    cv2.imwrite(os.path.join(save_path, 'im_black_offset.png'), im_black_offset)
    cv2.imwrite(os.path.join(save_path, 'im_fuse_offset2.png'), im_black_offset_2 + 0.5 * img_transed)
    cv2.imwrite(os.path.join(save_path, 'im_black_offset_fuse.png'), im_black_offset + 0.5 * img_transed)
    cv2.imwrite(os.path.join(save_path, 'im_black_pred_fuse.png'), im_black_pred + 0.5 * img_transed)
    cv2.imwrite(os.path.join(save_path, 'im_fuse.png'), im_black_pred + 0.5 * img_transed)
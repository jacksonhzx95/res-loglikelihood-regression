import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds, get_affine_transform, affine_transform
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box, get_center_scale


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
    # exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/exp/ce_regress18_FB_lr_ft-512x512_res18_ce_regress-flowB.yaml'
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/offline_work_place/ce_test'
    kpt_json = os.path.join(exp_path, 'test_gt_kpt.json')
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/face_landmark/'
    IMG_PREFIX = 'RawImage/TestAll'
    ANN = '400_senior'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    save_path = os.path.join(exp_path, 'val_visual_s')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    kpt_w, kpt_h = (256, 256)
    kpt_num = 19
    fontScale = 0.5

    # Blue color in BGR
    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Line thickness of 2 px
    thickness = 2
    for i in range(len(kpt_data)):
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        annoFolder = os.path.join(gt_path, img_name[:-4] + '.txt')
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

        # print(img[:, :, 0].max())
        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        # center_p, scale_p = get_center_scale(kpt_w/1.25, kpt_w/1.25, aspect_ratio=_aspect_ratio, scale_mult=1.0)
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)
        trans = get_affine_transform(center, scale, 0, [kpt_w, kpt_h])
        # img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        img = cv2.warpAffine(img, trans, (int(kpt_w), int(kpt_h)), flags=cv2.INTER_LINEAR)
        for j in range(kpt_num):

            # coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center_p, scale_p,
            #                              [kpt_w, kpt_h])
            coord_draw = np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])])
            gt_kpts = affine_transform(gt_kpt[j][0:2], trans)
            # pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            # gt_pts.append((int(gt_kpts[0]), int(gt_kpts[1])))
            # landmark_dist.append(abs(coord_draw[0] - gt_kpt[j][0] + (coord_draw[1] - gt_kpt[j][1])))

            # print(coord_draw)
            img = cv2.circle(img, (int(gt_kpts[0]),
                                   int(gt_kpts[1])),
                             radius=2, color=[0, 255, 0], thickness=2)
            img = cv2.circle(img, (int(coord_draw[0]), int(coord_draw[1])),
                             radius=2, color=[0, 255, 255], thickness=2)

            img = cv2.putText(img, f'L {j}', (int(gt_kpts[0]),
                                   int(gt_kpts[1])), font,
                                fontScale, color, thickness, cv2.LINE_AA)

        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '.png'), img)

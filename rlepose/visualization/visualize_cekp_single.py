import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box, get_center_scale
from rlepose.utils.presets import CETransform_beta


def scoliosis_pts_process(pts):
    joints_3d = np.zeros((len(pts), 2, 2), dtype=np.float32)
    for i in range(len(pts)):
        joints_3d[i, 0, 0] = pts[i][0]
        joints_3d[i, 1, 0] = pts[i][1]
        joints_3d[i, :2, 1] = 1
    return joints_3d


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
    # exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/exp/ce_env2s_22_06_Aug1-512x512_EfNets_ce_regress-flow.yaml'
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/offline_work_place/ce_test'
    kpt_json = os.path.join(exp_path, 'test_gt_kpt.json')
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/face_landmark/'
    IMG_PREFIX = 'RawImage/TestAll'
    ANN = '400_senior'
    transformation = CETransform_beta(scale_factor=0.4,
                                      input_size=[512, 512],
                                      output_size=[128, 128],
                                      flip=False,
                                      rot=30, sigma=0.3,
                                      train=True, loss_type='RLELoss', shift=[0.1, 0.1])

    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    save_path = os.path.join(exp_path, 'val_visual')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    kpt_w, kpt_h = (512, 512)
    kpt_num = 19
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
        gt_kpt = scoliosis_pts_process(gt_kpt)
        # read img
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)
        label = dict(joints=gt_kpt)
        label['width'] = img.shape[1]
        label['height'] = img.shape[0]
        target = transformation(img, label)
        gt_points = target['target_uv']
        gt_points = (gt_points + 0.5) * 512
        gt_points = gt_points.reshape(19, 2)
        img = target.pop('image')
        img = (img.numpy().transpose(1, 2, 0) + 0.5) * 255
        img = np.ascontiguousarray(img, dtype=np.uint8)
        for j in range(kpt_num):
            # coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
            #                              [kpt_w, kpt_h])
            # print(coord_draw)
            if int(gt_points[j][0]) > 512 or int(int(gt_points[j][1])) > 512:
                print(gt_points[j][0], gt_points[j][1])
                continue
            img = cv2.circle(img, (int(gt_points[j][0]),
                                   int(gt_points[j][1])),
                             radius=2, color=[0, 255, 0], thickness=2)
            # img = cv2.circle(img, (int(coord_draw[0]), int(coord_draw[1])),
            #                  radius=3, color=[0, 0, 255], thickness=3)

        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '.png'), img)

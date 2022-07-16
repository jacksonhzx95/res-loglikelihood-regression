import json
import cv2
import os
from scipy.io import loadmat
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import random
from rlepose.utils.transforms import transform_preds
from rlepose.utils.bbox import _box_to_center_scale, _center_scale_to_box, get_center_scale


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
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/offline_work_place/default'
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/train'
    ANN = 'labels/train'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    save_path = os.path.join(exp_path, 'val_visual', 'distribution')
    color_list = generate_color_list(100)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # kpt_file = open(kpt_json)
    # kpt_data = json.load(kpt_file)
    img_list = os.listdir(os.path.join(DATASET_PATH, IMG_PREFIX))
    kpt_w, kpt_h = (512, 1024)
    kpt_num = 68
    im_black = np.zeros((1024, 512, 3), dtype='uint8')
    im_black_sub = np.zeros((1024, 512, 3), dtype='uint8')
    point_list = np.zeros(shape=[kpt_num, 50, 2])
    for i in range(50):
        img_name = img_list[i]
        # kpt_coord = kpt_data[i]['keypoints']
        gt_img_ann = loadmat(os.path.join(gt_path, img_name))['p2']
        gt_kpt = rearrange_pts(gt_img_ann)
        # read img
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)
        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        center, scale = get_center_scale(kpt_w, kpt_h, aspect_ratio=_aspect_ratio, scale_mult=0.75)
        for j in range(kpt_num):
            # img = cv2.circle(img, (int((kpt_coord[j * 3 + 1]) * img_size[0] / img_kpt_size[1]), int(kpt_coord[j *
            # 3] * img_size[1] / img_kpt_size[0])), radius=3, color=[0, 0, 255], thickness=3) coord_draw_beta =
            # transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center_beta, scale_beta,
            # [kpt_w, kpt_h]) coord_draw = transform_preds(coord_draw_beta, center, scale, [kpt_w_beta, kpt_h_beta])
            # coord_draw_beta = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]),
            # center_beta, scale_beta, [kpt_w, kpt_h])
            coord_draw = transform_preds(np.array([gt_kpt[j][0], gt_kpt[j][1]]), center, scale,
                                         [img_w, img_h])
            # print(coord_draw)
            # gt_point_x = gt_kpt[j][0] * kpt_h / img_size[0]
            # gt_point_y = gt_kpt[j][1] * kpt_w / img_size[1]
            # im_black = cv2.circle(im_black, (int(gt_point_x),
            #                                  int(gt_point_y)),
            #                       radius=2, color=color_list[j], thickness=2)
            point_list[j, i, :] = coord_draw
            # im_black = cv2.circle(im_black, (int(coord_draw[0]),
            #                                  int(coord_draw[1])),
            #                       radius=2, color=color_list[j], thickness=2)

            # if j == 33:
            #     im_black_sub = cv2.circle(im_black_sub, (int(coord_draw[0]),
            #                                      int(coord_draw[1])),
            #                           radius=2, color=color_list[j], thickness=2)

            # img = cv2.circle(img, (int(coord_draw[0]), int(coord_draw[1])),
            #                  radius=3, color=[0, 0, 255], thickness=3)
    point_list = np.asarray(point_list)
    for kpt in range(10):
        x = point_list[kpt, :, 0]
        y = point_list[kpt, :, 1]
        deltaX = (max(x) - min(x)) / 10
        deltaY = (max(y) - min(y)) / 10

        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY

        print(xmin, xmax, ymin, ymax)

        # create meshgrid
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        # a = kernel(positions)
        # plt.matshow(a, cmap='jet')
        # dis_space = np.linspace(min(x))
        f = np.reshape(kernel(positions).T, xx.shape)
        plt.matshow(f, cmap='jet')

        # fig = plt.figure(figsize=(10, 20))
        # ax = fig.gca()
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymin)
        # cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        # cset = ax.contour(xx, yy, f, color='k')
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # plt.title('2D Ga')
        plt.savefig((str(kpt) + '.png'))
    cv2.imwrite(os.path.join(save_path, 'distribution.png'), im_black)
    cv2.imwrite(os.path.join(save_path, 'distribution_sub.png'), im_black_sub)

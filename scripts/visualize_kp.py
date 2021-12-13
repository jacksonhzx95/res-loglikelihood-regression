import json
import cv2
import os
from scipy.io import loadmat
import numpy as np
from rlepose.utils.transforms import transform_preds


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
    exp_path = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/exp/beta4' \
               '-1024x512_res50_scoliosic_regress-flow.yaml'
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
    kpt_w, kpt_h = (512, 1024)
    kpt_num = 68
    for i in range(len(kpt_data)):
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
            # img = cv2.circle(img, (int((kpt_coord[j * 3 + 1]) * img_size[0] / img_kpt_size[1]),
            #                        int(kpt_coord[j * 3] * img_size[1] / img_kpt_size[0])),
            #                  radius=3, color=[0, 0, 255], thickness=3)
            # coord_draw_beta = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center_beta,
            #                                   scale_beta, [kpt_w, kpt_h])
            # coord_draw = transform_preds(coord_draw_beta, center, scale,
            #                              [kpt_w_beta, kpt_h_beta])
            # coord_draw_beta = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center_beta,
            #                                   scale_beta, [kpt_w, kpt_h])
            coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
                                         [kpt_w, kpt_h])
            print(coord_draw)
            img = cv2.circle(img, (int(coord_draw[0]), int(coord_draw[1])),
                             radius=3, color=[0, 0, 255], thickness=3)
            img = cv2.circle(img, (int(gt_kpt[j][0]),
                                   int(gt_kpt[j][1])),
                             radius=3, color=[0, 255, 0], thickness=3)
        cv2.imwrite(os.path.join(save_path, img_name), img)
        # cv2.imwrite(
        #     '/mnt/sd2/Keypoint_detection/HRNet-Facial-Landmark-Detection/visualization_test/hm_' + image_path[-10:],
        #     heatmap)
        # np.expand_dims(target, axis=0)
        # preds = decode_preds(score_map, meta[
        # 'center'], meta['scale'], [64, 64])
        # print('target', target.size())
        # print('img', img.shape)
        # cor = torch.reshape(target, (1, 96, 256, 64))
        # # print(target.size())
        # # cor = target
        # tpts1 = tpts * 4
        # # cor1 = np.expand_dims(cor, axis=0)
        # cor = get_preds(cor)
        # cor = cor.cpu().numpy()
        # cor1 = cor[0] * 4
        # im = (img.transpose([1, 2, 0])) * 255
        # # im = im * 0.7
        # for j in range(len(cor1)):
        #     # pts[j]
        #     # txt = str(j)
        #     # print('(', cor1[j][0], ',', cor1[j][1], ')')
        #     im = cv2.circle(im, (int(tpts1[j][0]), int(tpts1[j][1])),
        #                     radius=10, color=[0, 0, 255], thickness=2)
        #     im = cv2.circle(im, (int(cor1[j][0]), int(cor1[j][1])),
        #                     radius=3, color=[0, 0, 255], thickness=3)
        # cv2.imwrite(
        #     '/mnt/sd2/Keypoint_detection/HRNet-Facial-Landmark-Detection/visualization_test/' + image_path[-10:],
        #     im)

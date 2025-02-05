"""Validation script."""
import logging
import os
import random
import sys
sys.path.insert(0, '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression')

import torch
import torch.multiprocessing as mp
from rlepose.models import builder
from rlepose.opt import cfg, opt
from rlepose.trainer_scoliosis import validate, validate_gt, validate_gt_3d
from rlepose.utils.env import init_dist
from rlepose.utils.transforms import get_coord, get_coord_scoliosis

num_gpu = torch.cuda.device_count()


def main():
    if opt.launcher in ['none', 'slurm']:
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):
    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    torch.backends.cudnn.benchmark = True

    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)

    m.cuda(opt.gpu)
    m = torch.nn.parallel.DataParallel(m, device_ids=[opt.gpu])

    output_3d = cfg.DATA_PRESET.get('OUT_3D', False)
    heatmap_to_coord = get_coord_scoliosis(cfg, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d)

    with torch.no_grad():
        if output_3d:
            err = validate_gt_3d(m, opt, cfg, heatmap_to_coord, opt.valid_batch)

            if opt.log:
                print('##### results: {} #####'.format(err))
        else:
            val_acc, mse, mape = validate_gt(m, opt, cfg, heatmap_to_coord, opt.valid_batch)
            # detbox_AP = validate(m, opt, cfg, heatmap_to_coord, opt.valid_batch)

            print(f'test acc: {val_acc} ; mse: {mse}; MAPE: {mape}#####')


if __name__ == "__main__":

    if opt.world_size > num_gpu:
        print(f'Wrong world size. Changing it from {opt.world_size} to {num_gpu}.')
        opt.world_size = num_gpu
    main()

import math

import torch
import torch.nn as nn

from .builder import LOSS


@LOSS.register_module
class MSELoss(nn.Module):
    ''' MSE Loss
    '''

    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, labels):
        pred_hm = output['heatmap']
        gt_hm = labels['target_hm']
        gt_hm_weight = labels['target_hm_weight']
        loss = 0.5 * self.criterion(pred_hm.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))
        return loss


@LOSS.register_module
class L1Loss(nn.Module):
    '''
    MSE Loss
    '''

    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, output, labels):
        pred_hm = output['heatmap']
        gt_hm = labels['target_hm']
        gt_hm_weight = labels['target_hm_weight']
        loss = 0.5 * self.criterion(pred_hm.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))

        return loss


@LOSS.register_module
class Focal_RLE_Loss(nn.Module):
    ''' RLE Regression Loss
        '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(Focal_RLE_Loss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)
        self.alpha = 0.25
        self.gamma = 2

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def log_wo_g(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]
        alpha_t = self.alpha * gt_uv + (1 - self.alpha) * (0.5 - gt_uv)
        pt = pred_jts * gt_uv + (0.5 - pred_jts) * (0.5 - gt_uv)

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = alpha_t * nf_loss * ((0.5 - pt) ** self.gamma) + Q_logprob
        else:
            Q_logprob = self.log_wo_g(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = 0.5 * nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()


@LOSS.register_module
class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def log_wo_g(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = 0.5 * nf_loss + Q_logprob
        else:
            Q_logprob = self.log_wo_g(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = 0.5 * nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()


@LOSS.register_module
class RLELoss3D(nn.Module):
    ''' RLE Regression Loss 3D
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss3D, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uvd'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)
        nf_loss = nf_loss * gt_uv_weight

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()


@LOSS.register_module
class RegressL1Loss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, ):
        super(RegressL1Loss, self).__init__()

        self.criterion = nn.MSELoss()

    def forward(self, output, labels):
        # nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        # nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        loss = self.criterion(gt_uv * gt_uv_weight, pred_jts * gt_uv_weight
                              )
        return loss.sum()

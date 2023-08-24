import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from utilities import SoftDiceLoss, mSoftDiceLoss, resize_segmentation, one_hot_embedding
from tensorboard_logger import log_value
import torch.nn.functional as F
import time
from common_test_Unet import nntestProstate, nntestATLAS, nntestCardiac

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR divided by 2 at 17th, 22th, 27th, 30th and 33th epochs"""
    # and more to converge
    lr = args.lr * (1 - epoch / args.epochs)**0.9
    
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def calculate_loss_origin(args, target_var, output, do_mixup = False):
    '''
    This is just an update function to make sure the calculated loss is identity to the original in some cases
    '''
    if args.deepsupervision:
        losssample = 0
        targetpicks = target_var.data.cpu().numpy()
        weights = np.array([1 / (2 ** i) for i in range(args.downsampling)])
        mask = np.array([True] + [True if i < args.downsampling - 1 else False for i in range(1, args.downsampling)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        for kds in range(args.downsampling):
            targetpickx = targetpicks[:, np.newaxis]
            s = np.ones(3) * 0.5 ** kds
            if args.ATLAS0Cardiac1Prostate2 == 1: # training with 128*128*8
                s[2] = 1
            axes = list(range(2, len(targetpickx.shape)))
            new_shape = np.array(targetpickx.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            if args.ATLAS0Cardiac1Prostate2 == 2: # training with 64*64*32
                if kds > 0:
                    new_shape[4] = new_shape[4] * 2
            new_shape = np.round(new_shape).astype(int)
            out_targetpickx = np.zeros(new_shape, dtype=targetpickx.dtype)
            for b in range(targetpickx.shape[0]):
                for c in range(targetpickx.shape[1]):
                    out_targetpickx[b, c] = resize_segmentation(targetpickx[b, c], new_shape[2:], order=0, cval=0)
            # if would be very slow if I used tensor from the begining.
            target_vars = torch.tensor(np.squeeze(out_targetpickx))

            if len(target_vars.size()) == 3:
                target_vars = target_vars.unsqueeze(0)

            target_vars = target_vars.long().cuda()
            target_vars = torch.autograd.Variable(target_vars)

            losssample += weights[kds] * imbalaned_loss_calculation(args, output[kds], target_vars, do_mixup)
    else:
        losssample = imbalaned_loss_calculation(args, output, target_var, do_mixup)
    return losssample

def imbalaned_loss_calculation(args, output: Tensor, target_var: Tensor, do_mixup = False) -> Tensor:
    '''
    output:     N x C x H x W x D
    target_var: N x H x W x D
    Here I assume C = 2, in other words, only for binary segmentation
    taken from https://github.com/ZerojumpLine/OverfittingUnderClassImbalance/
    '''
    inp = output
    target = target_var.long()
    num_classes = inp.size()[1]

    i0 = 1
    i1 = 2

    while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
        inp = inp.transpose(i0, i1)
        i0 += 1
        i1 += 1

    inp = inp.contiguous()
    inp = inp.view(-1, num_classes)

    target = target.view(-1,)

    # now inp is [N,C], target is [N,]
    y_one_hot = one_hot_embedding(target.data.cpu(), num_classes)

    # the loss calculations are taken from https://github.com/ZerojumpLine/OverfittingUnderClassImbalance/tree/master/Unet/nnunet/training/loss_functions
    ################# mix up #################
    if do_mixup:
        targetmix = target.flip(0)
        targetmix = targetmix.long()
        targetmix = targetmix.view(-1,)
        if args.asy:
            r = [0, 1]
            # I should consider the y_comb
            # this is the case, when this is any possible the the label of y should change
            rall = [i for i, e in enumerate(r) if e == 1]
            y_comb0 = target
            y_comb1 = targetmix
            # if the other one is taken as one of the rare classes, the combination should change
            for rindex in rall:
                y_comb0 = torch.where(targetmix == rindex, targetmix, y_comb0)
                y_comb1 = torch.where(target == rindex, target, y_comb1)

            # if there are several rare classes, we dont want to mix them
            # keep the y_comb as original labels, when they are rare classes.
            for rindex in rall:
                y_comb0 = torch.where(target == rindex, target, y_comb0)
                y_comb1 = torch.where(targetmix == rindex, targetmix, y_comb1)

            # if the another component mixup lambda is less than the margin, set as tumor
            if args.lam < args.margin:
                y_one_hot = one_hot_embedding(y_comb0.data.cpu(), num_classes)
            else:
                y_one_hot = one_hot_embedding(target.data.cpu(), num_classes)
            if 1-args.lam < args.margin:
                y_one_hotmix = one_hot_embedding(y_comb1.data.cpu(), num_classes)
            else:
                y_one_hotmix = one_hot_embedding(targetmix.data.cpu(), num_classes)
            y_one_hotmixup = args.lam * y_one_hot + (1 - args.lam) * y_one_hotmix
            # if the sample is generated by mixup, I do not want middle results, such as "0.3 tumor"
            # have no loss when background and kidney are mixed, or tumor is small
            # it might have problems, gradients of DSC are backwards, (it would be eventually taken as BG, which is not as expected.)
            ydsclossposition = torch.zeros(y_one_hotmixup.size()[0], dtype=torch.bool)
            for kcls in range(y_one_hotmixup.size()[1]):
                ydsclosspositionccls = (y_one_hotmixup[:, kcls] > 0) & (y_one_hotmixup[:, kcls] < 1)
                ydsclossposition = ydsclossposition.float() + ydsclosspositionccls.float()
            ydsclossposition = 1 - ydsclossposition # this would be 0/1, 0 indicates that there are portion which I dont want.
            y_one_hotmixup = torch.floor(y_one_hotmixup)
            y_one_hot = y_one_hotmixup
        else:
            y_one_hotmix = one_hot_embedding(targetmix.data.cpu(), num_classes)
            y_one_hot = args.lam * y_one_hot + (1 - args.lam) * y_one_hotmix
    
    y_one_hot = y_one_hot.cuda()
    ydsclossposition = torch.ones(y_one_hot.size()[0], dtype=torch.bool)
    ydsclossposition = torch.reshape(ydsclossposition, (output.shape[0], output.shape[2], output.shape[3], output.shape[4], 1))
    ydsclossposition = ydsclossposition.transpose(4, 3)
    ydsclossposition = ydsclossposition.transpose(3, 2)
    ydsclossposition = ydsclossposition.transpose(2, 1)
    ydsclossposition = ydsclossposition.float().cuda()

    ################# asymmetric large margin loss #################
    r = [0, 1]
    r = torch.reshape(torch.tensor(r), [1, len(r)])
    rRepeat = torch.cat(inp.shape[0] * [r])
    # this is the input to softmax, which will give us q
    inppost = inp - rRepeat.float().cuda() * y_one_hot * args.asy_margin
    
    # do the softmax and get q
    p_y_given_x_train = torch.softmax(inppost, 1)
    e1 = 1e-6  ## without the small margin, it would lead to nan after several epochs
    log_p_y_given_x_train = (p_y_given_x_train + e1).log()

    ################# asymmetric focal loss #################
    r = [0, 1]
    r = torch.reshape(torch.tensor(r), [1, len(r)])
    rRepeat = torch.cat(log_p_y_given_x_train.shape[0] * [r])

    focal_conduct_active = (1 - p_y_given_x_train + e1) ** args.asy_focal
    focal_conduct_inactive = torch.ones(p_y_given_x_train.size())

    focal_conduct = focal_conduct_active * (1-rRepeat.float().cuda()) + focal_conduct_inactive.cuda() * rRepeat.float().cuda()
    m_log_p_y_given_x_train = focal_conduct * log_p_y_given_x_train

    num_samples = m_log_p_y_given_x_train.size()[0]

    ce_loss = - (1. / num_samples) * m_log_p_y_given_x_train * y_one_hot
    ce_loss = ce_loss.sum()

    inpost = torch.reshape(inppost, (output.shape[0], output.shape[2], output.shape[3], output.shape[4], output.shape[1]))
    # then use transpose [B, H, W, D, C] to [B, C, H, W, D]
    inpost = inpost.transpose(4, 3)
    inpost = inpost.transpose(3, 2)
    inpost = inpost.transpose(2, 1)
    dc_loss = mSoftDiceLoss(inpost, target_var, focal_conduct, loss_mask = ydsclossposition)

    loss_func = dc_loss + ce_loss

    return loss_func

def validateATLAS(DatafileValFold, model, logging, epoch, Savename, args):
    model.eval()

    DSC, SENS, PREC = nntestATLAS(model, True, Savename + '/results/', False,
                        ImgsegmentSize=args.patch_size, 
                        deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSClesion', DSC[0], epoch)
        log_value('SENSlesion', SENS[0], epoch)
        log_value('PREClesion', PREC[0], epoch)
    return DSC.mean()

def validateProstate(DatafileValFold, model, logging, epoch, Savename, args):
    model.eval()

    DSC, SENS, PREC = nntestProstate(model, True, Savename + '/results/', False,
                        ImgsegmentSize=args.patch_size, 
                        deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSCprostate', DSC[0], epoch)
        log_value('SENSprostate', SENS[0], epoch)
        log_value('PRECprostate', PREC[0], epoch)
    return DSC.mean()

def validateCardiac(DatafileValFold, model, logging, epoch, Savename, args):
    model.eval()

    DSC, SENS, PREC = nntestCardiac(model, True, Savename + '/results/', False,
                        ImgsegmentSize=args.patch_size, 
                        deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold, NumsClass=4)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSCc1', DSC[0], epoch)
        log_value('SENSc1', SENS[0], epoch)
        log_value('PRECc1', PREC[0], epoch)
        log_value('DSCc2', DSC[1], epoch)
        log_value('SENSc2', SENS[1], epoch)
        log_value('PRECc2', PREC[1], epoch)
        log_value('DSCc3', DSC[2], epoch)
        log_value('SENSc3', SENS[2], epoch)
        log_value('PRECc3', PREC[2], epoch)
    return DSC.mean()


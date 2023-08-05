import torch
import numpy as np
import torch.nn as nn
from utilities import SoftDiceLoss
from tensorboard_logger import log_value
import torch.nn.functional as F
from utilities import resize_segmentation
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

def calculate_loss_origin(args, target_var, output, criterion = nn.CrossEntropyLoss().cuda()):
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
            losssample += weights[kds] * (criterion(output[kds], target_vars) + 
                    SoftDiceLoss(output[kds], target_vars, list(range(args.NumsClass))))
    else:
        losssample = SoftDiceLoss(output, target_var, list(range(args.NumsClass))) + criterion(output, target_var)
    return losssample


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


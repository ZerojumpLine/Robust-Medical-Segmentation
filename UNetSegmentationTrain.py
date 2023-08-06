import argparse
import os
import shutil
import time
import sys
import logging
import logging.config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
from Unet import Generic_UNet, InitWeights_He
from common_Unet import adjust_learning_rate, AverageMeter, calculate_loss_origin
# used for logging to TensorBoard
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from sampling_multiprocess import getbatch
from tensorboard_logger import configure, log_value
from utilities import save_checkpoint, accuracy
os.environ['KMP_WARNINGS'] = 'off'

parser = argparse.ArgumentParser(description='PyTorch nnU-Net Training')
# General configures.
parser.add_argument('--name', default='3DUnet', type=str, help='name of experiment')
parser.add_argument('--print-freq', '-p', default=40, type=int, help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# Training configures.
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=10, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--numIteration', default=700, type=int, help='num of iteration per epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--sgd0orAdam1orRms2', default=2, type=float, help='choose the optimizer')
parser.add_argument('--det', action='store_true', help='control seed to for control experiments')
# Network configures.
parser.add_argument('--maxsample', type=float, default=50, help='sample from cases, large number leads to longer time')
parser.add_argument('--evalevery', type=float, default=10, help='evaluation every epoches')
parser.add_argument('--downsampling', default=4, type=int, help='too see if I need deeper arch')
parser.add_argument('--features', default=30, type=int, help='feature maps for the entry level')
parser.add_argument('--deepsupervision', action='store_true', help='use deep supervision, just like nnunet')
parser.add_argument('--patch-size', default=[80,80,80], nargs='+', type=int, help='patch size')
# Dataset configures.
parser.add_argument('--ATLAS0Cardiac1Prostate2', default=0, type=float, help='choose the dataset')
args = parser.parse_args()
best_prec1 = 0

if args.det:
    np.random.seed(79)
    torch.manual_seed(79)
    torch.cuda.manual_seed_all(79)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    ## note that I can only control the sampling case, but not the sampling patches (it is controled by a global seed).
else:
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False

def main():
    global best_prec1
    
    ############################## some dataset specific configs. ##############################
    if args.ATLAS0Cardiac1Prostate2 == 0:
        dataset = 'ATLAS'
        # this is for brain lesion
        from common_Unet import validateATLAS as validate
        DatafileTrainqueueFold = './data/datafile/Dataset_Brain_lesion/GE 750 Discoverytraining/'
        DatafileValFold = './data/datafile/Dataset_Brain_lesion/GE 750 Discoverytest/'
        args.NumsInputChannel = 1
        args.NumsClass = 2
    if args.ATLAS0Cardiac1Prostate2 == 1:
        dataset = 'Cardiac'
        # this is for cardiac
        from common_Unet import validateCardiac as validate
        DatafileTrainqueueFold = './data/datafile/Dataset_Cardiac/1training/'
        DatafileValFold = './data/datafile/Dataset_Cardiac/1test/'
        args.NumsInputChannel = 1
        args.NumsClass = 4
    if args.ATLAS0Cardiac1Prostate2 == 2:
        dataset = 'Prostate'
        # this is for prostate
        from common_Unet import validateProstate as validate
        DatafileTrainqueueFold = './data/datafile/Dataset_Prostate/BMCtraining/'
        DatafileValFold = './data/datafile/Dataset_Prostate/BMCtest/'
        args.NumsInputChannel = 1
        args.NumsClass = 2

    ############################## init logging #########################
    Savename = args.name
    directory = "./output/%s/%s/"%(dataset, Savename)

    if not os.path.exists(directory):
        os.makedirs(directory)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(directory, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    # ignore the nibabel warnings.
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })

    if args.tensorboard: configure("./output/%s/%s"%(dataset, Savename))

    ############################## create model ##############################
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    conv_per_stage = 2
    base_num_features = args.features

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    net_num_pool_op_kernel_sizes = []
    if args.patch_size[0] == args.patch_size[1] == args.patch_size[2]:
        for kiter in range(0, args.downsampling):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 2])
    if args.ATLAS0Cardiac1Prostate2 == 1:
        for kiter in range(0, args.downsampling):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 1])
    if args.ATLAS0Cardiac1Prostate2 == 2:
        net_num_pool_op_kernel_sizes.append([2, 2, 1])
        for kiter in range(0, args.downsampling - 1):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 2])
    net_conv_kernel_sizes = []
    for kiter in range(0,args.downsampling+1) : # (0,6)
        net_conv_kernel_sizes.append([3,3,3])

    model = Generic_UNet(args.NumsInputChannel, base_num_features, args.NumsClass,
                                len(net_num_pool_op_kernel_sizes),
                                conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, args.deepsupervision, False, lambda x: x, InitWeights_He(1e-2),
                                net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    if args.sgd0orAdam1orRms2 == 0 :
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    if args.sgd0orAdam1orRms2 == 1 :
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=3e-5, amsgrad=True)
    if args.sgd0orAdam1orRms2 == 2 :
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-04, weight_decay=0.0001, momentum=0.6)

    # get the number of model parameters
    logging.info('Number of model parameters: {} MB'.format(sum([p.data.nelement() for p in model.parameters()])/1e6))

    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:' + str(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # when started, it needs initalization of prec1.
            prec1 = 0
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # multiprocess
    mp_pool = ThreadPool(processes=1)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if mp_pool is None:
            # sequence processing
            # sample more validation cases in one iteration
            sampling_results = getbatch(DatafileTrainqueueFold, args.batch_size, args.numIteration, args.maxsample, 
                                        logging, ImgsegmentSize=args.patch_size)

        elif epoch == args.start_epoch:  # Not previously submitted in case of first epoch
            # to get the sampling from the multiprocess. the sampling parameters might have mismatch
            # get one results.
            sampling_results = getbatch(DatafileTrainqueueFold, args.batch_size, args.numIteration, args.maxsample, 
                                        logging, ImgsegmentSize=args.patch_size)
            # sub new job.
            sampling_job = mp_pool.apply_async(getbatch, (DatafileTrainqueueFold, args.batch_size,
                                                          args.numIteration, args.maxsample,
                                                          logging, args.patch_size))
        elif epoch == args.epochs - 1:  # last iteration
            # do not need to submit job
            sampling_results = sampling_job.get()
            mp_pool.close()
            mp_pool.join()
        else:
            # get old job and submit new job
            sampling_results = sampling_job.get()
            ## otherwise it consumes a lot of memory
            sampling_job = mp_pool.apply_async(getbatch, (DatafileTrainqueueFold, args.batch_size,
                                                          args.numIteration, args.maxsample,
                                                          logging, args.patch_size))
        # input shape N, H, W, D, Channl
        # target shape N, H, W, D, Class
        inputnor = sampling_results[0]
        target = sampling_results[1]

        # train for one epoch
        train(inputnor, target, model, optimizer, epoch, logging, args)
        del inputnor
        del target

        # evaluate on validation set every 5 epoches
        if epoch % args.evalevery == 0 or epoch == args.epochs-1 :
            prec1 = validate(DatafileValFold, model, logging, epoch, Savename, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, dataset, Savename)
    logging.info('Best overall DSCuracy: %s ', best_prec1)



def train(inputnor, target, model, optimizer, epoch, logging, args):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for iteration in range(args.numIteration):
        targetpick = torch.tensor(target[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :])
        target_var = targetpick.long().cuda()
        target_var = torch.autograd.Variable(target_var)
        inputnorpick = torch.tensor(inputnor[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :, :])
        inputnor_var = inputnorpick.float().cuda()
        inputnor_var = torch.autograd.Variable(inputnor_var)

        # compute output
        output = model(inputnor_var)

        # folowing nnunet...
        outorigin = []
        for kds in range(args.downsampling):
            outtemp = output[kds]
            outorigin.append(outtemp[:, 0:args.NumsClass, :, : , :])
        loss = calculate_loss_origin(args, target_var, outorigin)

        optimizer.zero_grad()
        loss.backward()
        ## hyper-parameters from nnunet...
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()

        del inputnor_var
        del target_var

        # measure accuracy and record loss
        if args.deepsupervision:
            prec1 = accuracy(output[0].data, targetpick.long().cuda(), topk=(1,))[0]
        else:
            prec1 = accuracy(output.data, targetpick.long().cuda(), topk=(1,))[0]
        losses.update(loss.data.item(), inputnorpick.size()[0])
        top1.update(prec1.item(), inputnorpick.size()[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, iteration, args.numIteration, batch_time=batch_time,
                      loss=losses, top1=top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

if __name__ == '__main__':
    main()
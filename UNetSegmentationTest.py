#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from scipy import signal
import nibabel as nib
import numpy as np
import argparse
from Unet import Generic_UNet, InitWeights_He
import math
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter
from collections import OrderedDict
from numpy.linalg import inv
from utilities import SoftDiceLoss, resize_segmentation, _concat, _concatmodel, ComputMetric
from sampling_multiprocess import get_augment_par
from common_test_Unet import pad_nd_image, _compute_steps_for_sliding_window, getallbatch
import torch.nn.functional as F


# In[ ]:


def testmap(model, saveresults, name, pathname = None, ImgsegmentSize = [80, 160, 160], deepsupervision=False, 
    DatafileValFoldtr=None, DatafileValFoldts=None, tta=False, ttalist=[0], NumsClass = 2, ttalistprob=[1]):
    
    batch_size = 1
    NumsInputChannel = 1

    DatafiletsFold = DatafileValFoldts
    DatafiletsImgc1 = DatafiletsFold + 'Imgpre-eval.txt'
    DatafiletsLabel = DatafiletsFold + 'seg-eval.txt'
    DatafiletsMask = DatafiletsFold + 'mask-eval.txt'

    Imgfiletsc1 = open(DatafiletsImgc1)
    Imgreadtsc1 = Imgfiletsc1.read().splitlines()
    if os.path.isfile(DatafiletsMask):
        Masktsfile = open(DatafiletsMask)
        Masktsread = Masktsfile.read().splitlines()
        Maskread = Masktsread
    Labeltsfile = open(DatafiletsLabel)
    Labeltsread = Labeltsfile.read().splitlines()

    Imgreadc1 = Imgreadtsc1
    Labelread = Labeltsread
    
    DSClist = []
    
    for numr in range(len(Imgreadc1)):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        
        if os.path.isfile(DatafiletsMask):
            Maskname = Maskread[numr]
            Maskload = nib.load(Maskname)
            roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()
        
        Imgc1 = np.float32(Imgc1)

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-2]

        channels = Imgc1[None, ...]

        from common_test_Unet import tta_rolling
        hp_results = tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision)

        ## use the mask to constratin the results
        if os.path.isfile(DatafiletsMask):
            PredSegmentationWithinRoi = hp_results * roi_mask
        else:
            PredSegmentationWithinRoi = hp_results
        # PredSegmentationWithinRoi = predSegmentation

        predlabel = np.argmax(PredSegmentationWithinRoi, axis=0)

        if NumsClass == 2:
            labelwt = gtlabel == 1
            predwt = predlabel == 1
    
            DSCwt, SENSwt, PRECwt = ComputMetric(labelwt, predwt)
    
            # print(DSCwt)
            DSClist.append([DSCwt])
        else:
            labelc1 = gtlabel == 1
            predc1 = predlabel == 1
            labelc2 = gtlabel == 2
            predc2 = predlabel == 2
            labelc3 = gtlabel == 3
            predc3 = predlabel == 3
    
            DSCc1, SENSc1, PRECc1 = ComputMetric(labelc1, predc1)
            DSCc2, SENSc2, PRECc2 = ComputMetric(labelc2, predc2)
            DSCc3, SENSc3, PRECc3 = ComputMetric(labelc3, predc3)
    
            # print(DSCwt)
            DSClist.append([DSCc1, DSCc2, DSCc3])
        
        for kcls in range(PredSegmentationWithinRoi.shape[0]):

            imgToSave = PredSegmentationWithinRoi[kcls, :, :, :]

            if saveresults:
                npDtype = np.dtype(np.float32)
                hdr_origin = Imgloadc1.header
                affine_origin = Imgloadc1.affine
                
                newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
                newLabelImg.set_data_dtype(npDtype)

                dimsImgToSave = len(imgToSave.shape)
                newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
                if len(newZooms) < dimsImgToSave:  # Eg if original image was 3D, but I need to save a multi-channel image.
                    newZooms = newZooms + [1.0] * (dimsImgToSave - len(newZooms))
                newLabelImg.header.set_zooms(newZooms)

                directory = "./output/%s/%s/" % (pathname, name)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                savename = directory + 'pred_' + kname + 'cls' + str(kcls) + '_prob.nii.gz'
                nib.save(newLabelImg, savename)
    
    DSCmean = np.array(DSClist)
    DSCmean = DSCmean.mean(axis=0)
    print(DSCmean)


# In[ ]:


prostateckpt = './checkpoints_seg/Prostate/3DUNet_vanilla_Prostate_det/checkpoint.pth.tar'
atlasckpt = './checkpoints_seg/ATLAS/3DUNet_vanilla_ATLAS_det/checkpoint.pth.tar'
atlasckpt_ci1 = './checkpoints_seg/ATLAS/3DUNet_asymargin_1_ATLAS_det/checkpoint.pth.tar'
atlasckpt_ci2 = './checkpoints_seg/ATLAS/3DUNet_asyfocal_4_ATLAS_det/checkpoint.pth.tar'
atlasckpt_rl1 = './checkpoints_seg/ATLAS/3DUNet_mixup_ATLAS_det/checkpoint.pth.tar'
atlasckpt_rl2 = './checkpoints_seg/ATLAS/3DUNet_GIN_ATLAS_det/checkpoint.pth.tar'
cardiacckpt = './checkpoints_seg/Cardiac/3DUNet_vanilla_Cardiac_det/checkpoint.pth.tar'


# In[ ]:


prostatevalpath = './data/datafile/Dataset_Prostate/BMCval/'
prostatetestpath = './data/datafile/Dataset_Prostate/BMCtest/'
prostatetestpath1 = './data/datafile/Dataset_Prostate/BIDMC/'
prostatetestpath2 = './data/datafile/Dataset_Prostate/HK/'
prostatetestpath3 = './data/datafile/Dataset_Prostate/I2CVB/'
prostatetestpath4 = './data/datafile/Dataset_Prostate/RUNMC/'
prostatetestpath5 = './data/datafile/Dataset_Prostate/UCL/'
prostatetestpaths = [prostatetestpath1, prostatetestpath2, prostatetestpath3, prostatetestpath4, prostatetestpath5]
#
atlasvalpath = './data/datafile/Dataset_Brain_lesion/Siemens Trioval/'
atlastestpath = './data/datafile/Dataset_Brain_lesion/Siemens Triotest/'
atlastestpath1 = './data/datafile/Dataset_Brain_lesion/GE 750 Discovery/'
atlastestpath2 = './data/datafile/Dataset_Brain_lesion/GE Signa Excite/'
atlastestpath3 = './data/datafile/Dataset_Brain_lesion/GE Signa HD-X/'
atlastestpath4 = './data/datafile/Dataset_Brain_lesion/Philips/'
atlastestpath5 = './data/datafile/Dataset_Brain_lesion/Philips Achieva/'
atlastestpath6 = './data/datafile/Dataset_Brain_lesion/Siemens Allegra/'
atlastestpath7 = './data/datafile/Dataset_Brain_lesion/Siemens Magnetom Skyra/'
atlastestpath8 = './data/datafile/Dataset_Brain_lesion/Siemens Prisma/'
atlastestpath9 = './data/datafile/Dataset_Brain_lesion/Siemens Skyra/'
atlastestpath10 = './data/datafile/Dataset_Brain_lesion/Siemens Sonata/'
atlastestpath11 = './data/datafile/Dataset_Brain_lesion/Siemens TrioTim/'
atlastestpath12 = './data/datafile/Dataset_Brain_lesion/Siemens Verio/'
atlastestpath13 = './data/datafile/Dataset_Brain_lesion/Siemens Vision/'
atlastestpaths = [atlastestpath1, atlastestpath2, atlastestpath3, atlastestpath4, atlastestpath5, 
                 atlastestpath6, atlastestpath7, atlastestpath8, atlastestpath9, atlastestpath10, 
                 atlastestpath11, atlastestpath12, atlastestpath13]
#
cardiacvalpath = './data/datafile/Dataset_Cardiac/1val/'
cardiactestpath = './data/datafile/Dataset_Cardiac/1test/'
cardiactestpath1 = './data/datafile/Dataset_Cardiac/2/'
cardiactestpath2 = './data/datafile/Dataset_Cardiac/3/'
cardiactestpath3 = './data/datafile/Dataset_Cardiac/4/'
cardiactestpath4 = './data/datafile/Dataset_Cardiac/5/'
cardiactestpaths = [cardiactestpath1, cardiactestpath2, cardiactestpath3, cardiactestpath4]


# # Prostate

# In[ ]:


savename = 'prostateval'
savedckpt = prostateckpt
patch_size = [64, 64, 32]
testlist = [0]
testprob = [1]
NumsInputChannel = 1
NumsClass = 2
pathname = 'prostate'
DatafileValFoldtr = None
DatafileValFoldts = prostatevalpath
#
args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 0}


# In[ ]:


# create model
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
conv_per_stage = 2
base_num_features = args['features']
args['features'] = base_num_features

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_num_pool_op_kernel_sizes = []
net_num_pool_op_kernel_sizes.append([2, 2, 1])
for kiter in range(0, args['downsampling'] - 1):  # (0,5)
    net_num_pool_op_kernel_sizes.append([2, 2, 2])
net_conv_kernel_sizes = []
for kiter in range(0, args['downsampling'] + 1):  # (0,6)
    net_conv_kernel_sizes.append([3, 3, 3])

model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                     len(net_num_pool_op_kernel_sizes),
                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                     dropout_op_kwargs,
                     net_nonlin, net_nonlin_kwargs, args['deepsupervision'], False, lambda x: x, InitWeights_He(1e-2),
                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
model = model.cuda()
model.eval()


# In[ ]:


torch.cuda.set_device(args['gpu'])
if args['resume']:
    if os.path.isfile(args['resume']):
        print("=> loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'], map_location='cuda:' + str(args['gpu']))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


# In[ ]:


testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


## test condition x 83
DatafileValFoldts = prostatetestpath
for k in tqdm(range(1,84)):
    savename = 'prostatetestsyn_' + str(k) 
    testlist = [k]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


# test on 5 other domains
testlist = [0]
for prostatetestpath in tqdm(prostatetestpaths):
    DatafileValFoldts = prostatetestpath
    savename = 'prostatetest_' + prostatetestpath.split('/')[-2]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# # Cardiac

# In[ ]:


savename = 'cardiacval'
savedckpt = cardiacckpt
patch_size = [128, 128, 8]
testlist = [0]
testprob = [1]
NumsInputChannel = 1
NumsClass = 4
pathname = 'cardiac'
DatafileValFoldtr = None
DatafileValFoldts = cardiacvalpath
#
args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 0}


# In[ ]:


# create model
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
conv_per_stage = 2
base_num_features = args['features']
args['features'] = base_num_features

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_num_pool_op_kernel_sizes = []
for kiter in range(0, args['downsampling']):  # (0,5)
    net_num_pool_op_kernel_sizes.append([2, 2, 1])
net_conv_kernel_sizes = []
for kiter in range(0, args['downsampling'] + 1):  # (0,6)
    net_conv_kernel_sizes.append([3, 3, 3])

model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                     len(net_num_pool_op_kernel_sizes),
                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                     dropout_op_kwargs,
                     net_nonlin, net_nonlin_kwargs, args['deepsupervision'], False, lambda x: x, InitWeights_He(1e-2),
                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
model = model.cuda()
model.eval()


# In[ ]:


torch.cuda.set_device(args['gpu'])
if args['resume']:
    if os.path.isfile(args['resume']):
        print("=> loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'], map_location='cuda:' + str(args['gpu']))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


# In[ ]:


testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


## test condition x 83
DatafileValFoldts = cardiactestpath
for k in tqdm(range(1,84)):
    savename = 'cardiactestsyn_' + str(k) 
    testlist = [k]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


# test on 4 other domains
testlist = [0]
for cardiactestpath in tqdm(cardiactestpaths):
    DatafileValFoldts = cardiactestpath
    savename = 'cardiactest_' + cardiactestpath.split('/')[-2]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# # ATLAS

# In[ ]:


savename = 'atlasval'
savedckpt = atlasckpt
patch_size = [128, 128, 128]
testlist = [0]
testprob = [1]
NumsInputChannel = 1
NumsClass = 2
pathname = 'atlas'
DatafileValFoldtr = None
DatafileValFoldts = atlasvalpath
#
args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 0}


# In[ ]:


torch.cuda.set_device(args['gpu'])
# create model
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
conv_per_stage = 2
base_num_features = args['features']
args['features'] = base_num_features

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_num_pool_op_kernel_sizes = []
for kiter in range(0, args['downsampling']):  # (0,5)
    net_num_pool_op_kernel_sizes.append([2, 2, 2])
net_conv_kernel_sizes = []
for kiter in range(0, args['downsampling'] + 1):  # (0,6)
    net_conv_kernel_sizes.append([3, 3, 3])

model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                     len(net_num_pool_op_kernel_sizes),
                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                     dropout_op_kwargs,
                     net_nonlin, net_nonlin_kwargs, args['deepsupervision'], False, lambda x: x, InitWeights_He(1e-2),
                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
model = model.cuda()
model.eval()

if args['resume']:
    if os.path.isfile(args['resume']):
        print("=> loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'], map_location='cuda:' + str(args['gpu']))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


# In[ ]:


testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


## test condition x 83
for k in tqdm(range(1,84)):
    savename = 'atlastestcondition_' + str(k) 
    testlist = [k]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 10, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


# test on other domains
testlist = [0]
for atlastestpath in tqdm(atlastestpaths):
    DatafileValFoldts = atlastestpath
    savename = 'atlastest_' + atlastestpath.split('/')[-2]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# ## ATLAS, ci_1

# In[ ]:


savename = 'atlasval'
savedckpt = atlasckpt_ci1
patch_size = [128, 128, 128]
testlist = [0]
testprob = [1]
NumsInputChannel = 1
NumsClass = 2
pathname = 'atlas_ci1'
DatafileValFoldtr = None
DatafileValFoldts = atlasvalpath
#
args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 0}


# In[ ]:


torch.cuda.set_device(args['gpu'])
# create model
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
conv_per_stage = 2
base_num_features = args['features']
args['features'] = base_num_features

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_num_pool_op_kernel_sizes = []
for kiter in range(0, args['downsampling']):  # (0,5)
    net_num_pool_op_kernel_sizes.append([2, 2, 2])
net_conv_kernel_sizes = []
for kiter in range(0, args['downsampling'] + 1):  # (0,6)
    net_conv_kernel_sizes.append([3, 3, 3])

model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                     len(net_num_pool_op_kernel_sizes),
                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                     dropout_op_kwargs,
                     net_nonlin, net_nonlin_kwargs, args['deepsupervision'], False, lambda x: x, InitWeights_He(1e-2),
                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
model = model.cuda()
model.eval()

if args['resume']:
    if os.path.isfile(args['resume']):
        print("=> loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'], map_location='cuda:' + str(args['gpu']))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


# In[ ]:


testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


## test condition x 83
for k in tqdm(range(1,84)):
    savename = 'atlastestcondition_' + str(k) 
    testlist = [k]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 10, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


# test on other domains
testlist = [0]
for atlastestpath in tqdm(atlastestpaths):
    DatafileValFoldts = atlastestpath
    savename = 'atlastest_' + atlastestpath.split('/')[-2]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# ## ATLAS, ci_2

# In[ ]:


savename = 'atlasval'
savedckpt = atlasckpt_ci2
patch_size = [128, 128, 128]
testlist = [0]
testprob = [1]
NumsInputChannel = 1
NumsClass = 2
pathname = 'atlas_ci2'
DatafileValFoldtr = None
DatafileValFoldts = atlasvalpath
#
args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 0}


# In[ ]:


torch.cuda.set_device(args['gpu'])
# create model
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
conv_per_stage = 2
base_num_features = args['features']
args['features'] = base_num_features

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_num_pool_op_kernel_sizes = []
for kiter in range(0, args['downsampling']):  # (0,5)
    net_num_pool_op_kernel_sizes.append([2, 2, 2])
net_conv_kernel_sizes = []
for kiter in range(0, args['downsampling'] + 1):  # (0,6)
    net_conv_kernel_sizes.append([3, 3, 3])

model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                     len(net_num_pool_op_kernel_sizes),
                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                     dropout_op_kwargs,
                     net_nonlin, net_nonlin_kwargs, args['deepsupervision'], False, lambda x: x, InitWeights_He(1e-2),
                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
model = model.cuda()
model.eval()

if args['resume']:
    if os.path.isfile(args['resume']):
        print("=> loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'], map_location='cuda:' + str(args['gpu']))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


# In[ ]:


testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


## test condition x 83
for k in tqdm(range(1,84)):
    savename = 'atlastestcondition_' + str(k) 
    testlist = [k]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 10, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


# test on other domains
testlist = [0]
for atlastestpath in tqdm(atlastestpaths):
    DatafileValFoldts = atlastestpath
    savename = 'atlastest_' + atlastestpath.split('/')[-2]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# ## ATLAS, rl_1

# In[ ]:


savename = 'atlasval'
savedckpt = atlasckpt_rl1
patch_size = [128, 128, 128]
testlist = [0]
testprob = [1]
NumsInputChannel = 1
NumsClass = 2
pathname = 'atlas_rl1'
DatafileValFoldtr = None
DatafileValFoldts = atlasvalpath
#
args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 0}


# In[ ]:


torch.cuda.set_device(args['gpu'])
# create model
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
conv_per_stage = 2
base_num_features = args['features']
args['features'] = base_num_features

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_num_pool_op_kernel_sizes = []
for kiter in range(0, args['downsampling']):  # (0,5)
    net_num_pool_op_kernel_sizes.append([2, 2, 2])
net_conv_kernel_sizes = []
for kiter in range(0, args['downsampling'] + 1):  # (0,6)
    net_conv_kernel_sizes.append([3, 3, 3])

model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                     len(net_num_pool_op_kernel_sizes),
                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                     dropout_op_kwargs,
                     net_nonlin, net_nonlin_kwargs, args['deepsupervision'], False, lambda x: x, InitWeights_He(1e-2),
                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
model = model.cuda()
model.eval()

if args['resume']:
    if os.path.isfile(args['resume']):
        print("=> loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'], map_location='cuda:' + str(args['gpu']))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


# In[ ]:


testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


## test condition x 83
for k in tqdm(range(1,84)):
    savename = 'atlastestcondition_' + str(k) 
    testlist = [k]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 10, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


# test on other domains
testlist = [0]
for atlastestpath in tqdm(atlastestpaths):
    DatafileValFoldts = atlastestpath
    savename = 'atlastest_' + atlastestpath.split('/')[-2]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# ## ATLAS, rl_2

# In[ ]:


savename = 'atlasval'
savedckpt = atlasckpt_rl2
patch_size = [128, 128, 128]
testlist = [0]
testprob = [1]
NumsInputChannel = 1
NumsClass = 2
pathname = 'atlas_rl2'
DatafileValFoldtr = None
DatafileValFoldts = atlasvalpath
#
args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 0}


# In[ ]:


torch.cuda.set_device(args['gpu'])
# create model
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
conv_per_stage = 2
base_num_features = args['features']
args['features'] = base_num_features

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_num_pool_op_kernel_sizes = []
for kiter in range(0, args['downsampling']):  # (0,5)
    net_num_pool_op_kernel_sizes.append([2, 2, 2])
net_conv_kernel_sizes = []
for kiter in range(0, args['downsampling'] + 1):  # (0,6)
    net_conv_kernel_sizes.append([3, 3, 3])

model = Generic_UNet(NumsInputChannel, base_num_features, NumsClass,
                     len(net_num_pool_op_kernel_sizes),
                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                     dropout_op_kwargs,
                     net_nonlin, net_nonlin_kwargs, args['deepsupervision'], False, lambda x: x, InitWeights_He(1e-2),
                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
model = model.cuda()
model.eval()

if args['resume']:
    if os.path.isfile(args['resume']):
        print("=> loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'], map_location='cuda:' + str(args['gpu']))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


# In[ ]:


testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


## test condition x 83
for k in tqdm(range(1,84)):
    savename = 'atlastestcondition_' + str(k) 
    testlist = [k]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 10, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


# In[ ]:


# test on other domains
testlist = [0]
for atlastestpath in tqdm(atlastestpaths):
    DatafileValFoldts = atlastestpath
    savename = 'atlastest_' + atlastestpath.split('/')[-2]
    args = {'resume': savedckpt, 'name': savename, 'saveresults': True, 'patch_size': patch_size, 'downsampling': 4,
       'ttalist': testlist, 'ttalistprob': testprob, 'features': 30, 'deepsupervision': True, 'gpu': 1}
    testmap(model, args['saveresults'], args['name'] + '/results/', pathname=pathname,
                    ImgsegmentSize=args['patch_size'], deepsupervision=args['deepsupervision'],
                    DatafileValFoldtr=DatafileValFoldtr, DatafileValFoldts=DatafileValFoldts, NumsClass=NumsClass,
                    ttalist=args['ttalist'], ttalistprob=args['ttalistprob'])


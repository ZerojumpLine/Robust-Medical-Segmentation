import os
import numpy as np
import nibabel as nib
import math
import random
import signal
from itertools import repeat
import multiprocessing as mp
from utilities import get_patch_size
from augmentImagerng import AugmenterAffineParams
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

# np.random.seed(12345)
# random.seed(12345)
## take image as 37*37*37, and the target as 21*21*21

def getbatch(DatafileFold, batch_size, iteration, maximumcase, logging, ImgsegmentSize=[80,80,80]):

    # LabelsegmentSize = 21
    # ImgsegmentSize = LabelsegmentSize + 2 * offset

    LabelsegmentSize = ImgsegmentSize

    samplenum = batch_size * iteration

    DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
    DatafileLabel = DatafileFold + 'seg-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()

    samplingpool = np.minimum(int(maximumcase), len(Imgreadc1))
    samplingpool = np.minimum(samplingpool, samplenum)
    samplefromeachcase = np.floor(samplenum / samplingpool)
    residualcase = int(samplenum - samplefromeachcase * samplingpool)

    if iteration > 100: # try not to print for meta validation...
        logging.info('Sampling training segments from ' + str(samplingpool) + ' cases')
        logging.info('Total patches are ' + str(samplenum))

    samplefromthiscase = np.ones((samplingpool)) * samplefromeachcase
    for numiter in range(residualcase):
        samplefromthiscase[numiter] = samplefromthiscase[numiter] + 1

    kimg = list(range(samplingpool))
    mp_pool = mp.Pool(processes=np.minimum(8, len(kimg)))
    mp_pool.daemon = False
    # print(os.getpid())
    # mp_pool.daemon = True

    caselist = np.random.randint(0, len(Imgreadc1), samplingpool)
    logging.info('Sampling caselist for training: ' + str(caselist))
    seednum = np.random.randint(0, 1e6)
    # print(caselist)

    try:
        with mp_pool as pool:
            results = pool.starmap(getsampleskitsatlas,
                                   zip(kimg, samplefromthiscase[kimg], repeat(Imgreadc1), repeat(Labelread), 
                                       repeat(ImgsegmentSize), repeat(LabelsegmentSize), caselist[kimg], repeat(seednum)))


    except mp.TimeoutError:
        print("time out?")
    except:  # Catches everything, even a sys.exit(1) exception.
        mp_pool.terminate()
        mp_pool.join()
        raise Exception("Unexpected error.")
    else:  # Nothing went wrong
        # Needed in case any processes are hanging. mp_pool.close() does not solve this.

        batchxnor = []
        batchy = []
        for knum in range(samplingpool):
            batchxnor.append(results[knum][0])
            batchy.append(results[knum][1])

        # I should wait here and terminate.
        mp_pool.close()
        mp_pool.terminate()
        mp_pool.join()

    batchxnor = np.vstack(batchxnor)
    batchy = np.vstack(batchy)
    listr = list(range(samplenum))
    np.random.shuffle(listr)
    batchxnor = batchxnor[listr, :, :, :, :]
    batchy = batchy[listr, :, :, :].astype('int')

    # batchy = one_hot_embedding(batchy, 4)

    return batchxnor, batchy, listr, None

def getsampleskitsatlas(kimg, samplefromthiscase, Imgreadc1, Labelread,
                        ImgsegmentSize, LabelsegmentSize, numr, seednum):

    np.random.seed(seednum + kimg)
    
    Imgnamec1 = Imgreadc1[numr]
    Imgloadc1 = nib.load(Imgnamec1)
    Imgc1 = Imgloadc1.get_fdata()
    Labelname = Labelread[numr]
    Labelload = nib.load(Labelname)
    gt_lbl_img = Labelload.get_fdata()

    roi_mask = np.ones(gt_lbl_img.shape)

    channels = Imgc1[None, ...] ## add one dimension

    batchxnor, batchy, numlist = getsamples(channels, gt_lbl_img, roi_mask, samplefromthiscase,
                    ImgsegmentSize, LabelsegmentSize)

    return batchxnor, batchy, numr, numlist

def getsamples(channels, gt_lbl_img, roi_mask, samplefromthiscase, ImgsegmentSize, LabelsegmentSize):

    # local_state = np.random.RandomState()
    '''
    Caution: For nnunet, fabian use np.random in the augmentation code
    I should generate different seed specifically for this (numpy)
    What I do here is very easy and stupid, setting seed for numpy based on the local_state
    '''
    # np.random.seed(np.random.randint(0, 1e6))

    range_x, range_y, range_z = roi_mask.shape

    
    Imgenlarge = np.zeros((1, max(ImgsegmentSize[0],range_x), max(ImgsegmentSize[1],range_y), max(ImgsegmentSize[2],range_z)))
    Maskenlarge = np.zeros((max(ImgsegmentSize[0],range_x), max(ImgsegmentSize[1],range_y), max(ImgsegmentSize[2],range_z)))
    Labelenlarge = np.zeros((max(ImgsegmentSize[0],range_x), max(ImgsegmentSize[1],range_y), max(ImgsegmentSize[2],range_z)))
    Imgenlarge[:, 0:range_x, 0:range_y, 0:range_z] = channels
    Labelenlarge[0:range_x, 0:range_y, 0:range_z] = gt_lbl_img
    Maskenlarge[0:range_x, 0:range_y, 0:range_z] = roi_mask

    batchxnor = np.zeros((int(samplefromthiscase), 1, ImgsegmentSize[0], ImgsegmentSize[1], ImgsegmentSize[2]))

    batchy = np.zeros((int(samplefromthiscase), LabelsegmentSize[0], LabelsegmentSize[1], LabelsegmentSize[2]))
    numlist = np.zeros((int(samplefromthiscase)))

    # cls wise sampling would lead to werid results..
    # numofcls = int(np.max(gt_lbl_img)) + 1
    '''
    I notice nnunet just use FG/BG sampling, maybe I should follow his implementation to get the same results.
    '''
    # just use FG / BG sampling
    numofcls = 2
    # I should divide samplefromthiscase into 3 pieces: BG, FG, residuals
    # therefore I just need two point
    samplingposition = []
    samplingposition.append(int(0))
    # this should be random chose the uppr or lower case.
    # otherwise, there would be a lot more BG
    FGportion = 0.5
    if np.random.uniform() > 0.5:
        samplingposition.append(int(np.ceil(samplefromthiscase*(1 - FGportion))))
    else:
        samplingposition.append(int(np.floor(samplefromthiscase*(1 - FGportion))))
    samplingposition.append(int(samplefromthiscase))

    # cls wise sampling...
    gt_lbl_img_inside_mask = Labelenlarge * Maskenlarge
    offsetx = [int(ImgsegmentSize[0] / 2), int(ImgsegmentSize[1] / 2), int(ImgsegmentSize[2] / 2)]
    lbx = offsetx[0]
    ubx = -offsetx[0]
    lby = offsetx[1]
    uby = -offsetx[1]
    lbz = offsetx[2]
    ubz = -offsetx[2]
    if gt_lbl_img_inside_mask.shape[0] == ImgsegmentSize[0]:
        lbx = int(ImgsegmentSize[0] / 2)
        ubx = int(ImgsegmentSize[0] / 2) + 1
    if gt_lbl_img_inside_mask.shape[1] == ImgsegmentSize[1]:
        lby = int(ImgsegmentSize[1] / 2)
        uby = int(ImgsegmentSize[1] / 2) + 1
    if gt_lbl_img_inside_mask.shape[2] == ImgsegmentSize[2]:
        lbz = int(ImgsegmentSize[2] / 2)
        ubz = int(ImgsegmentSize[2] / 2) + 1

    gt_lbl_img_inside_mask = gt_lbl_img_inside_mask[lbx:ubx, lby:uby,lbz:ubz]
    Maskenlargemask = Maskenlarge[lbx:ubx, lby:uby,lbz:ubz]

    if samplefromthiscase >= numofcls:  # I have enough samples for different cls inside one image, it is for normal training, sample 1000patches or more.

        for kcls in range(0, numofcls):

            if kcls == 0:
                kclschoice = 0
                bgcls_mask = (gt_lbl_img_inside_mask == 0) * Maskenlargemask
                Labelindex = bgcls_mask.nonzero()
            else:
                kclschoice = np.random.randint(1, np.max((int(np.max(gt_lbl_img)+1), 2)))
                Labelindex = np.where(gt_lbl_img_inside_mask == kclschoice)
            Labelindex_x = Labelindex[0]
            Labelindex_y = Labelindex[1]
            Labelindex_z = Labelindex[2]
            if len(Labelindex_x) == 0:
                ## it can be not Gt on the given slice
                ## find it near the center
                Labelindex_all = np.where(Labelenlarge == kclschoice)
                if len(Labelindex_all[0]) == 0:  # no cls in this map
                    Labelindex_all = np.where(Maskenlarge > 0)
                Labelindex_all = list(Labelindex_all)
                Labelindex_all[0] = np.mean(Labelindex_all[0]) - int(ImgsegmentSize[0] / 2)
                Labelindex_all[1] = np.mean(Labelindex_all[1]) - int(ImgsegmentSize[1] / 2)
                Labelindex_all[2] = np.mean(Labelindex_all[2]) - int(ImgsegmentSize[2] / 2)
                Labelindex_x = [min(max(int(Labelindex_all[0]), 0), int(Labelenlarge.shape[0] - ImgsegmentSize[0]))]
                Labelindex_y = [min(max(int(Labelindex_all[1]), 0), int(Labelenlarge.shape[1] - ImgsegmentSize[1]))]
                Labelindex_z = [min(max(int(Labelindex_all[2]), 0), int(Labelenlarge.shape[2] - ImgsegmentSize[2]))]

            startpos = samplingposition[kcls]
            endpos = samplingposition[kcls+1]

            for k in range(startpos, endpos):  # sampling from different classes
                # print(Labelindex_x.size)
                numindex = np.random.randint(0, len(Labelindex_x))
                # numindex = np.minimum(k, Labelindex_x.size - 1)
                # this line would lead to error when multi process of cas xxxx816/817 with augmentation
                # ...np.random.randint(0, Labelindex_x.size-1)
                # I dont need -1, anyway. remove -1, it would be fine?
                # maybe the operation -1 could lead to something werid.
                selindex_x = Labelindex_x[numindex]
                selindex_y = Labelindex_y[numindex]
                selindex_z = Labelindex_z[numindex]
                ## selindex is the rightmost pixel.


                channs_of_sample_per_path, lbls_predicted_part_of_sample = getaugmentpath(Imgenlarge, Labelenlarge,
                                                                                          selindex_x, selindex_y,
                                                                                          selindex_z, ImgsegmentSize,
                                                                                          LabelsegmentSize)

                batchxnor[int(k), :, :, :, :] = channs_of_sample_per_path[0]
                batchy[int(k), :, :, :] = lbls_predicted_part_of_sample

    else:  # I do not have enough samples inside this imsage, sample randomly for each caes. it can be slow for sampling a lot
        # it is for the meta-training sampling.

        # cls wise sampling...
        bgcls_mask = (gt_lbl_img_inside_mask == 0) * Maskenlargemask

        for k in range(int(samplefromthiscase)):  #
            # I should make
            if np.random.uniform() > FGportion :
                kcls = 0
            else:
                kcls = np.random.randint(1, np.max((int(np.max(gt_lbl_img)+1), 2)))

            if kcls == 0:
                Labelindex = bgcls_mask.nonzero()
            else:
                Labelindex = np.where(gt_lbl_img_inside_mask == kcls)  ## it would take much time.
                # the process is fundementally slow? for example, gt_lbl == 1 would also take long

            Labelindex_x = Labelindex[0]
            Labelindex_y = Labelindex[1]
            Labelindex_z = Labelindex[2]
            if len(Labelindex_x) == 0:
                ## it can be not Gt on the given slice
                ## find it near the center
                Labelindex_all = np.where(Labelenlarge == kcls)
                if len(Labelindex_all[0]) == 0:  # no cls in this map
                    Labelindex_all = np.where(Maskenlarge > 0)
                Labelindex_all = list(Labelindex_all)
                Labelindex_all[0] = np.mean(Labelindex_all[0]) - int(ImgsegmentSize[0] / 2)
                Labelindex_all[1] = np.mean(Labelindex_all[1]) - int(ImgsegmentSize[1] / 2)
                Labelindex_all[2] = np.mean(Labelindex_all[2]) - int(ImgsegmentSize[2] / 2)
                Labelindex_x = [min(max(int(Labelindex_all[0]), 0), int(Labelenlarge.shape[0] - ImgsegmentSize[0]))]
                Labelindex_y = [min(max(int(Labelindex_all[1]), 0), int(Labelenlarge.shape[1] - ImgsegmentSize[1]))]
                Labelindex_z = [min(max(int(Labelindex_all[2]), 0), int(Labelenlarge.shape[2] - ImgsegmentSize[2]))]


            # print(Labelindex_x.size)
            numindex = np.random.randint(0, len(Labelindex_x))
            selindex_x = Labelindex_x[numindex]
            selindex_y = Labelindex_y[numindex]
            selindex_z = Labelindex_z[numindex]

            channs_of_sample_per_path, lbls_predicted_part_of_sample = getaugmentpath(Imgenlarge, Labelenlarge,
                                                                                      selindex_x, selindex_y,
                                                                                      selindex_z, ImgsegmentSize,
                                                                                      LabelsegmentSize)

            batchxnor[int(k), :, :, :, :] = channs_of_sample_per_path[0]
            batchy[int(k), :, :, :] = lbls_predicted_part_of_sample

    return batchxnor, batchy, numlist

def getaugmentpath(Imgenlarge, Labelenlarge, selindex_x, selindex_y, selindex_z, ImgsegmentSize,
                   LabelsegmentSize):
    """
    - I change the augmentation process to the nnunet implementation.
    - I do the sampling myself, but connect the augmentor of batchgenerator.
    """

    # ok, my patch now is x * y * z, however, nnunet assume z * y * x
    # I make a transpose here to make it consistent.

    Imgenlarge = np.transpose(Imgenlarge, (0, 3, 2, 1))
    Labelenlarge = np.transpose(Labelenlarge, (2, 1, 0))
    tmp = selindex_z
    selindex_z = selindex_x
    selindex_x = tmp
    ImgsegmentSizet = ImgsegmentSize.copy()
    tmp = ImgsegmentSizet[2]
    ImgsegmentSizet[2] = ImgsegmentSizet[0]
    ImgsegmentSizet[0] = tmp
    LabelsegmentSizet = LabelsegmentSize.copy()
    tmp = LabelsegmentSizet[2]
    LabelsegmentSizet[2] = LabelsegmentSizet[0]
    LabelsegmentSizet[0] = tmp

    channs_of_sample_per_path = []

    # try to make it exactly like DM's implementation
    coord_center = np.zeros(3, dtype=int)
    coord_center[0] = int(selindex_x + ImgsegmentSizet[0] // 2)
    coord_center[1] = int(selindex_y + ImgsegmentSizet[1] // 2)
    coord_center[2] = int(selindex_z + ImgsegmentSizet[2] // 2)

    if 1 > 0:  # image level. totally 15.

        ## I should replace my augmentor with nnaugmentor here.
        ## initialize params
        params = {
            "selected_data_channels": None,
            "selected_seg_channels": [0],

            "do_elastic": False,
            "elastic_deform_alpha": (0., 900.),
            "elastic_deform_sigma": (9., 13.),
            "p_eldef": 0.2,

            "do_scaling": True,
            "scale_range": (0.7, 1.4),
            "independent_scale_factor_for_each_axis": False,
            "p_scale": 0.2,

            "do_rotation": True,
            "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            "rotation_p_per_axis": 1,
            "p_rot": 0.2,

            "random_crop": False,
            "random_crop_dist_to_border": None,

            "do_gamma": True,
            "gamma_retain_stats": True,
            "gamma_range": (0.7, 1.5),
            "p_gamma": 0.3,

            "do_mirror": True,
            "mirror_axes": (0, 1, 2),

            "dummy_2D": False,
            "mask_was_used_for_normalization": False,
            "border_mode_data": "constant",

            "all_segmentation_labels": None,  # used for cascade
            "move_last_seg_chanel_to_data": False,  # used for cascade
            "cascade_do_cascade_augmentations": False,  # used for cascade
            "cascade_random_binary_transform_p": 0.4,
            "cascade_random_binary_transform_p_per_label": 1,
            "cascade_random_binary_transform_size": (1, 8),
            "cascade_remove_conn_comp_p": 0.2,
            "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
            "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

            "do_additive_brightness": False,
            "additive_brightness_p_per_sample": 0.15,
            "additive_brightness_p_per_channel": 0.5,
            "additive_brightness_mu": 0.0,
            "additive_brightness_sigma": 0.1,

            "num_threads": 12,
            "num_cached_per_thread": 1,
        }

        # cut a large img patch.

        samplekernal_primary = 1
        half_LabelSize_primary = LabelsegmentSizet.copy()
        new_patch_size = get_patch_size(LabelsegmentSizet, params['rotation_x'], params['rotation_y'], params['rotation_z'], (0.85, 1.25))

        ## the rot_x, rot_y and rot_z are literatually in 80 * 160 * 160, which is the patch size
        ## it is nothing to do with the global axis
        ## in my implementation here, the parameters are corresponding to
        # rot x - > longitudinal axis, rot y - > frontal axis, rot z - > sagittal axis

        half_LabelSize_primary[0] = int(new_patch_size[0]) // 2
        half_LabelSize_primary[1] = int(new_patch_size[1]) // 2
        half_LabelSize_primary[2] = int(new_patch_size[2]) // 2
        ## more here
        lbls_of_sample_primary = Getimagepatchwithcoord(Labelenlarge[np.newaxis, ...], half_LabelSize_primary,
                                                        samplekernal_primary, coord_center[0],
                                                        coord_center[1], coord_center[2])
        lbls_of_sample_primary = lbls_of_sample_primary.squeeze()

        # context, fetch a larger context
        samplekernal_sub2 = 1
        half_ImgsegmentSize_sub2 = LabelsegmentSizet.copy()
        half_ImgsegmentSize_sub2[0] = half_LabelSize_primary[0]
        half_ImgsegmentSize_sub2[1] = half_LabelSize_primary[1]
        half_ImgsegmentSize_sub2[2] = half_LabelSize_primary[2]
        channs_of_sample_sub2 = Getimagepatchwithcoord(Imgenlarge, half_ImgsegmentSize_sub2,
                                                       samplekernal_primary, coord_center[0],
                                                       coord_center[1], coord_center[2])

        order_data = 3
        order_seg = 1
        border_val_seg = -1
        patch_size = np.zeros(3, dtype=int)
        patch_size = ImgsegmentSizet

        tr_transforms = []

        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size, patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg,
            order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
        ))

        # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
        # channel gets in the way
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=0.1))  # inverted gamma

        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                               p_per_sample=params["p_gamma"]))

        if params.get("do_mirror") or params.get("mirror"):
            tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        tr_transforms.append(RenameTransform('seg', 'target', True))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)

        item = dict()
        item['data'] = channs_of_sample_sub2[np.newaxis, ...]
        item['seg'] = lbls_of_sample_primary[np.newaxis, np.newaxis, ...]
        item = tr_transforms(**item)
        channels_augment = item['data'].numpy()
        gt_lbl_img_augment = item['target'].numpy()

        ImagetoSample = channels_augment[0, :, :, :, :]
        LbltoSample = gt_lbl_img_augment[0, 0, :, : , :]

    ## change back to x * y * z
    ImagetoSample = np.transpose(ImagetoSample, (0, 3, 2, 1))
    LbltoSample = np.transpose(LbltoSample, (2, 1, 0))

    channs_of_sample_per_path.append(ImagetoSample)
    ## Label
    lbls_predicted_part_of_sample = LbltoSample

    return channs_of_sample_per_path, lbls_predicted_part_of_sample

def Getimagepatchwithcoord(Imgenlarge, half_ImgsegmentSize_sub1, samplekernal, xcentercoordinate, ycentercoordinate, zcentercoordinate):
    xleftlist = np.arange(xcentercoordinate, -1, -samplekernal)
    if len(xleftlist) > half_ImgsegmentSize_sub1[0]:
        xleftlist = xleftlist[1:half_ImgsegmentSize_sub1[0] + 1]
    else:
        xleftlist = xleftlist[1:]
    xrightlist = np.arange(xcentercoordinate, Imgenlarge.shape[1], samplekernal)
    if len(xrightlist) > half_ImgsegmentSize_sub1[0]:
        xrightlist = xrightlist[1:half_ImgsegmentSize_sub1[0] + 1]
    else:
        xrightlist = xrightlist[1:]
    xcoordinatelist = np.concatenate([xleftlist[::-1], [xcentercoordinate], xrightlist])
    xleftpadding = half_ImgsegmentSize_sub1[0] - len(xleftlist)
    xrightpadding = half_ImgsegmentSize_sub1[0] - len(xrightlist)
    # for y direction
    yleftlist = np.arange(ycentercoordinate, -1, -samplekernal)
    if len(yleftlist) > half_ImgsegmentSize_sub1[1]:
        yleftlist = yleftlist[1:half_ImgsegmentSize_sub1[1] + 1]
    else:
        yleftlist = yleftlist[1:]
    yrightlist = np.arange(ycentercoordinate, Imgenlarge.shape[2], samplekernal)
    if len(yrightlist) > half_ImgsegmentSize_sub1[1]:
        yrightlist = yrightlist[1:half_ImgsegmentSize_sub1[1] + 1]
    else:
        yrightlist = yrightlist[1:]
    ycoordinatelist = np.concatenate([yleftlist[::-1], [ycentercoordinate], yrightlist])
    yleftpadding = half_ImgsegmentSize_sub1[1] - len(yleftlist)
    yrightpadding = half_ImgsegmentSize_sub1[1] - len(yrightlist)
    # for z direction
    zleftlist = np.arange(zcentercoordinate, -1, -samplekernal)
    if len(zleftlist) > half_ImgsegmentSize_sub1[2]:
        zleftlist = zleftlist[1:half_ImgsegmentSize_sub1[2] + 1]
    else:
        zleftlist = zleftlist[1:]
    zrightlist = np.arange(zcentercoordinate,Imgenlarge.shape[3], samplekernal)
    if len(zrightlist) > half_ImgsegmentSize_sub1[2]:
        zrightlist = zrightlist[1:half_ImgsegmentSize_sub1[2] + 1]
    else:
        zrightlist = zrightlist[1:]
    zcoordinatelist = np.concatenate([zleftlist[::-1], [zcentercoordinate], zrightlist])
    zleftpadding = half_ImgsegmentSize_sub1[2] - len(zleftlist)
    zrightpadding = half_ImgsegmentSize_sub1[2] - len(zrightlist)

    channs_of_sample_per_path = Imgenlarge[:, np.min(xcoordinatelist):np.max(xcoordinatelist) + 1:samplekernal,
                                     np.min(ycoordinatelist):np.max(ycoordinatelist) + 1:samplekernal,
                                     np.min(zcoordinatelist):np.max(zcoordinatelist) + 1:samplekernal]
    # pad x
    channs_of_sample_per_path = np.concatenate((np.zeros(
        (channs_of_sample_per_path.shape[0], int(xleftpadding), channs_of_sample_per_path.shape[2],
         channs_of_sample_per_path.shape[3])), channs_of_sample_per_path,
                                                np.zeros((channs_of_sample_per_path.shape[0],
                                                          int(xrightpadding),
                                                          channs_of_sample_per_path.shape[2],
                                                          channs_of_sample_per_path.shape[3]))),
        axis=1)
    # pad y
    channs_of_sample_per_path = np.concatenate((np.zeros(
        (channs_of_sample_per_path.shape[0], channs_of_sample_per_path.shape[1], int(yleftpadding),
         channs_of_sample_per_path.shape[3])), channs_of_sample_per_path,
                                                np.zeros((channs_of_sample_per_path.shape[0],
                                                          channs_of_sample_per_path.shape[1],
                                                          int(yrightpadding),
                                                          channs_of_sample_per_path.shape[3]))),
        axis=2)
    # pad z
    channs_of_sample_per_path = np.concatenate((np.zeros(
        (channs_of_sample_per_path.shape[0], channs_of_sample_per_path.shape[1],
         channs_of_sample_per_path.shape[2], int(zleftpadding))), channs_of_sample_per_path,
                                                np.zeros((channs_of_sample_per_path.shape[0],
                                                          channs_of_sample_per_path.shape[1],
                                                          channs_of_sample_per_path.shape[2],
                                                          int(zrightpadding)))), axis=3)
    return channs_of_sample_per_path


def calc_border_int_of_3d_img(img_3d):
    border_int = np.mean([img_3d[0, 0, 0],
                          img_3d[-1, 0, 0],
                          img_3d[0, -1, 0],
                          img_3d[-1, -1, 0],
                          img_3d[0, 0, -1],
                          img_3d[-1, 0, -1],
                          img_3d[0, -1, -1],
                          img_3d[-1, -1, -1]
                          ])
    return border_int

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = np.eye(num_classes)  # [D,D]

    return y[labels]  # [N,D]

def get_augment_par():
    augm_img_prms_tr = {'origin': None}
    augm_img_prms_tr['origin'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.})
    augm_img_prms_tr['scaling1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.05})
    augm_img_prms_tr['scaling2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.15})
    augm_img_prms_tr['scaling3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.25})
    augm_img_prms_tr['scaling4'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.35})
    augm_img_prms_tr['scaling5'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.45})
    augm_img_prms_tr['rotFrontal_x1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 5.), 'scaling': 0.})
    augm_img_prms_tr['rotFrontal_x2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 15.), 'scaling': 0.})
    augm_img_prms_tr['rotFrontal_x3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 25.), 'scaling': 0.})
    augm_img_prms_tr['rotSagittal_y1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 5., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotSagittal_y2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 15., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotSagittal_y3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 25., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotLongitudinal_z1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (5., 0., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotLongitudinal_z2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (15., 0., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotLongitudinal_z3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (25., 0., 0.), 'scaling': 0.})

    augm_sample_prms_tr = {'mirror1': None}
    augm_sample_prms_tr['mirror1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (1., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['mirror2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 1., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['mirror3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 1.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Frontal_x1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 1., '180': 0., '270': 1.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Frontal_x2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 1., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Sagittal_y1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 1., '180': 0., '270': 1.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Sagittal_y2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 1., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Longitudinal_z1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 1., '180': 0., '270': 1.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Longitudinal_z2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 1., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0.1, 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0.3, 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0.5, 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma1invert'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.1}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma2invert'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.3}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma3invert'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.5}, 'simulowres': {'zoom': 1.}}   
    augm_sample_prms_tr['brightnessadd1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0.05, 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessadd2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0.15, 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessadd3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0.25, 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}                          
    augm_sample_prms_tr['brightnessmul1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0.05, 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessmul2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0.15, 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessmul3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0.25, 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['contrast1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.05}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['contrast2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.15}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['contrast3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.25}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['blur1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.5, 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['blur2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.7, 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['blur3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.9, 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['sharpen1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.9, 'sharpen': True},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['sharpen2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.7, 'sharpen': True},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['sharpen3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.5, 'sharpen': True},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['noise1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.025}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['noise2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.075}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['noise3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.125}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['simulowres1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 0.9}}
    augm_sample_prms_tr['simulowres2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 0.7}}
    augm_sample_prms_tr['simulowres3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 0.5}}  
    ## this is only for the default test-time augmentation...mirror in three directions
    augm_sample_prms_tr['mirror123'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (1., 1., 1.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    return augm_img_prms_tr, augm_sample_prms_tr
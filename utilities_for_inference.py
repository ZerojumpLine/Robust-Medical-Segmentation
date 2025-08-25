import SimpleITK as sitk
from typing import Union, Tuple, List
import numpy as np
import collections
import torch
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, gaussian_laplace

# helper functions copy pasted
def resample_by_res(mov_img_obj, new_spacing, interpolator = sitk.sitkLinear, logging = True):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_img_obj.GetDirection())
    resample.SetOutputOrigin(mov_img_obj.GetOrigin())
    resample.SetUseNearestNeighborExtrapolator(True)
    mov_spacing = mov_img_obj.GetSpacing()

    resample.SetOutputSpacing(new_spacing)
    RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
    new_size = np.array(mov_img_obj.GetSize()) *  RES_COE 

    resample.SetSize( [int(sz+1) for sz in new_size] )
    if logging:
        print("Spacing: {} -> {}".format(mov_spacing, new_spacing))
        print("Size {} -> {}".format( mov_img_obj.GetSize(), new_size ))

    return resample.Execute(mov_img_obj)

def resample_lb_by_shape(mov_lb_obj, new_shape, interpolator=sitk.sitkLinear, ref_img=None, logging=True):
    """
    Resample label image to a specific shape.
    
    Args:
        mov_lb_obj: Input label image (SimpleITK Image)
        new_shape: Desired output shape as tuple (z, y, x)
        interpolator: Interpolation method (default: sitk.sitkLinear)
        ref_img: Reference image for copying information (optional)
        logging: Enable logging output
    
    Returns:
        Resampled label image
    """
    # Get current image properties
    current_size = mov_lb_obj.GetSize()  # SimpleITK order: (x, y, z)
    current_spacing = mov_lb_obj.GetSpacing()  # SimpleITK order: (x, y, z)
    
    # Convert new_shape from (z, y, x) to SimpleITK order (x, y, z)
    new_shape_sitk = (new_shape[2], new_shape[1], new_shape[0])
    
    # Calculate new spacing based on desired shape
    # spacing = (physical_size) / (voxel_count)
    new_spacing = [
        current_size[0] * current_spacing[0] / new_shape_sitk[0],
        current_size[1] * current_spacing[1] / new_shape_sitk[1],
        current_size[2] * current_spacing[2] / new_shape_sitk[2]
    ]
    
    if logging:
        print(f"Current size (x,y,z): {current_size}")
        print(f"Current spacing (x,y,z): {current_spacing}")
        print(f"Target shape (z,y,x): {new_shape}")
        print(f"Target shape (x,y,z): {new_shape_sitk}")
        print(f"Calculated new spacing (x,y,z): {new_spacing}")
    
    # Extract label values
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)  # numpy order: (z, y, x)
    lbvs = np.unique(src_mat)
    
    if logging:
        print("Label values: {}".format(lbvs))
    
    # Initialize output volume with zeros of the target shape
    out_vol = np.zeros(new_shape, dtype=np.int32)
    
    # Process each label separately
    for idx, lbv in enumerate(lbvs):
        if lbv == 0:  # Skip background
            continue
            
        if logging:
            print(f"Processing label {lbv}")
            
        _src_curr_mat = np.float32(src_mat == lbv) 
        _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
        _src_curr_obj.CopyInformation(mov_lb_obj)
        
        # Resample with calculated spacing
        _tar_curr_obj = resample_by_res(_src_curr_obj, new_spacing, interpolator, logging)
        _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv
        
        # Add current label to output volume
        out_vol[_tar_curr_mat == lbv] = lbv
    
    # Create output image
    out_obj = sitk.GetImageFromArray(out_vol)
    out_obj.SetSpacing(new_spacing)
    
    # Set origin and direction to match input
    out_obj.SetOrigin(mov_lb_obj.GetOrigin())
    out_obj.SetDirection(mov_lb_obj.GetDirection())
    
    if ref_img is not None:
        out_obj.CopyInformation(ref_img)
    
    return out_obj

def resample_by_res(image, new_spacing, interpolator=sitk.sitkLinear, logging=True):
    """
    Helper function to resample image by resolution (spacing)
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    
    if logging:
        print(f"Resampling from {original_size} to {new_size}")
        print(f"Spacing from {original_spacing} to {new_spacing}")
    
    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(image)

def tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision):
    hp_results = 0

    augm_img_prms_tr, augm_sample_prms_tr = get_augment_par()
    augms_prms = {**augm_img_prms_tr, **augm_sample_prms_tr}

    for ttaindex, ttaindexprob in zip(ttalist, ttalistprob):

        print('processing tta index: ', ttaindex)
        
        ## I shall do the augmentation here, for the full image.
        ## maybe I should re-implement the inverse transform here.
        ## mainly for the spatial augmentation, methods in augmentImage.py.

        ## it should have 51 components.
        '''be careful, it is not the same as the paper notion, it is the order with sampling_multipreprocess.py'''
        augmentIDlist = [[0], [1, -2], [3, -4], [5, -6], [7, -8], [9, -10], \
                    [11, -12], [13, -14], [15, -16], [17, -18], [19, -20], [21, -22], [23, -24],[25, -26], [27, -28], \
                    [29], [30], [31], [32, -33], [34], [35, -36], [37], [38, -39], [40], \
                    [41, -42], [43, -44], [45, -46], [47, -48], [49, -50], [51, -52], \
                    [53, -54], [55, -56], [57, -58], [59, -60], [61, -62], [63, -64], [65, -66], [67, -68], [69, -70], \
                    [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83]]

        kcount = 0
        skipflag = False
        for key in augms_prms:
            for augmentIDs in augmentIDlist[kcount]:
                if ttaindex == np.abs(augmentIDs):
                    prmssel = augms_prms[key]
                    if augmentIDs > 0:
                        rng = 1
                    else:
                        rng = -1
                    skipflag = True
                    break
            if skipflag:
                break
            else:
                kcount += 1

        if ttaindex < 29 and ttaindex > 0:

            Imgsize_origin = channels.shape[1:]
            
            rotrange_z = prmssel['rot_xyz'][2]
            rotrange_y = prmssel['rot_xyz'][1]
            rotrange_x = prmssel['rot_xyz'][0]
            scalingrange = prmssel['scaling']
            
            new_patch_size_primary = get_patch_size(Imgsize_origin, 
                (-rotrange_z / 360 * 2. * np.pi, rotrange_z / 360 * 2. * np.pi), 
                (-rotrange_y / 360 * 2. * np.pi, rotrange_y / 360 * 2. * np.pi),
                (-rotrange_x / 360 * 2. * np.pi, rotrange_x / 360 * 2. * np.pi), 
                (1/(1+scalingrange), 1+scalingrange))

            ## return the center? to get the right transform.
            (channels_augment, _, transf_mtx) = augment_imgs_of_case(channels.copy(), None,
                                                            None, None, prmssel, new_patch_size_primary, rng)
            transf_mtx_inv = inv(transf_mtx)
            # Img = []
            # Img.append(channels[0, :, :, 87])
            # Img.append(channels_augment[0, :, :, 87])
            # Img.append(channels_augment_revert[0, :, :, 87])
            # show_threeimg(Img)
            # Img = []
            # Img.append(gtlabel[:, :, 87])
            # Img.append(gt_lbl_img_augment_revert[:, :, 87])
            # Img.append(gt_lbl_img_augment_revert[:, :, 87]-gtlabel[:, :, 87])
            # show_threeimg(Img)
            # print(np.sum(np.abs(gtlabel)))
            # print(np.sum(np.abs(gt_lbl_img_augment_revert)))
            # print(np.sum(np.abs(gtlabel-gt_lbl_img_augment_revert)))
        else:

            '''
            I do not need to do anything for anisotropy patches, because it operats with the whole images.
            '''

            channels_per_path = []
            channels_per_path.append(channels.copy())
            
            Imgenlargeref = channels[:, ::4, ::4, ::4]

            (channs_of_sample_per_path, _) = augment_sample(channels_per_path,
                                                        None, prmssel, Imgenlargeref, rng, 0)
            channels_augment = channs_of_sample_per_path[0]

            '''debug: visualization'''
            # Img = []
            # Img.append(gtlabel[:, :, 87])
            # Img.append(lbls_predicted_part_of_sample[:, :, 87])
            # Img.append(lbls_predicted_part_of_sample_augment[:, :, 87])
            # show_threeimg(Img)

        offset = [0, 0, 0]

        pad_border_mode = 'constant'
        pad_kwargs = dict()
        pad_kwargs['constant_values'] = 0
        data, slicer = pad_nd_image(channels_augment, ImgsegmentSize, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape
        step_size = 0.5
        steps = _compute_steps_for_sliding_window(ImgsegmentSize, data_shape[1:], step_size)

        hp = np.zeros([NumsClass] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([NumsClass] + list(data.shape[1:]), dtype=np.float32)
        xpixels = []
        ypixels = []
        zpixels = []

        for jx in steps[0]:
            for jy in steps[1]:
                for jz in steps[2]:
                    xpixels.append(jx)
                    ypixels.append(jy)
                    zpixels.append(jz)

        inputxnor = getallbatch(data, ImgsegmentSize, xpixels, ypixels, zpixels, offset)

        ## gaussian filter
        patch_size = [ImgsegmentSize[0], ImgsegmentSize[1], ImgsegmentSize[2]]
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * 1. / 8 for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        inputxnor = torch.tensor(np.array(inputxnor))
        inputxnor = inputxnor.float().cuda()
        for xlist in range(0, len(inputxnor), batch_size):
            batchxnor = inputxnor[xlist: xlist + batch_size, :, :, :, :]

            xstarts = xpixels[xlist: xlist + batch_size]
            ystarts = ypixels[xlist: xlist + batch_size]
            zstarts = zpixels[xlist: xlist + batch_size]

            with torch.no_grad():
                if tta:
                    auglist = [0, 1, 2, 3, 4, 5, 6, 7]
                    # auglist = [0, 4]
                    num_results = len(auglist)
                    mirror_axes = [0, 1, 2]
                    for m in range(num_results):
                        if m == 0:
                            pred = model(batchxnor)
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output = 1 / num_results * pred[0]
                            else:
                                output = 1 / num_results * pred
                        if m == 1 and (2 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4,)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4,))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4,))
                        if m == 2 and (1 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (3,)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (3,))
                            else:
                                output += 1 / num_results * torch.flip(pred, (3,))
                        if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4, 3)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4, 3))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4, 3))
                        if m == 4 and (0 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (2,)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (2,))
                            else:
                                output += 1 / num_results * torch.flip(pred, (2,))
                        if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4, 2)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4, 2))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4, 2))
                        if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (2, 3)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (2, 3))
                            else:
                                output += 1 / num_results * torch.flip(pred, (2, 3))
                        if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4, 3, 2)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4, 3, 2))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4, 3, 2))
                else:
                    pred = model(batchxnor)
                    ## in case it is multi-task model.
                    if len(pred) == 2:
                        pred = pred[0]
                    if deepsupervision:
                        output = pred[0]
                    else:
                        output = pred
            output = output.data.cpu().numpy()

            kbatch = 0
            for xstart, ystart, zstart in zip(xstarts, ystarts, zstarts):
                # only crop the center parts.
                # maybe I should use gaussain? to do ..
                # hp[:, xstart + offset:xstart + offset + PredSizetest, ystart + offset:ystart + offset + PredSizetest,
                # zstart + offset:zstart + offset + PredSizetest] = output[kbatch, :, offset:offset + PredSizetest,
                #                                                   offset:offset + PredSizetest,
                #                                                   offset:offset + PredSizetest]
                hp[:, xstart:xstart + ImgsegmentSize[0], ystart:ystart + ImgsegmentSize[1],
                zstart:zstart + ImgsegmentSize[2]] += output[kbatch, :, :, :, :] * gaussian_importance_map
                aggregated_nb_of_predictions[:, xstart:xstart + ImgsegmentSize[0], ystart:ystart + ImgsegmentSize[1],
                zstart:zstart + ImgsegmentSize[2]] += gaussian_importance_map
                kbatch = kbatch + 1

        slicer = tuple(
            [slice(0, hp.shape[i]) for i in
             range(len(hp.shape) - (len(slicer) - 1))] + slicer[1:])
        hp = hp[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
        hp = hp / aggregated_nb_of_predictions

        ## to see if the probability map needs revert spatial transformations 
        hp_revert = hp
        if ttaindex < 29 and ttaindex > 0:
            ## it is the revert transform.
            (hp_revert, _, _) = augment_imgs_of_case(hp.copy(), None,
            None, None, prmssel, Imgsize_origin, -rng, transf_mtx_inv)
        
        if (ttaindex >= 29 and ttaindex <= 40) or ttaindex == 83:
            hp_per_path = []
            hp_per_path.append(hp)
            (hp_revert_per_path, _) = augment_sample(hp_per_path,
                                                        None, prmssel, np.zeros(1), -rng, 0)
            hp_revert = hp_revert_per_path[0]
        
        ## hp_revert with shape: 1, D, W, H
        ## Here I want to try to ensemble with prediction score.
        # hp_revert = np.exp(hp_revert) / np.sum(np.exp(hp_revert), axis=0)
        hp_results += hp_revert / sum(ttalistprob) * ttaindexprob
    
    return hp_results


def getallbatch(Imgenlarge, ImgsegmentSize, xpixels, ypixels, zpixels, offset):

    inputxnor = []

    # normal pathway
    for (selindex_x, selindex_y, selindex_z) in zip(xpixels, ypixels, zpixels):
        coord_center = np.zeros(3, dtype=int)
        coord_center[0] = selindex_x + ImgsegmentSize[0] // 2
        coord_center[1] = selindex_y + ImgsegmentSize[1] // 2
        coord_center[2] = selindex_z + ImgsegmentSize[2] // 2

        samplekernal_primary = 1
        channs_of_sample_per_path_normal = Imgenlarge[:,
                                    coord_center[0] - ImgsegmentSize[0] // 2: coord_center[0] + ImgsegmentSize[0] // 2,
                                    coord_center[1] - ImgsegmentSize[1] // 2: coord_center[1] + ImgsegmentSize[1] // 2,
                                    coord_center[2] - ImgsegmentSize[2] // 2: coord_center[2] + ImgsegmentSize[2] // 2]
        inputxnor.append(channs_of_sample_per_path_normal)

    return inputxnor

def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer

def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

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

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        # it should consider both directions, I think.
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, -rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, -rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, -rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, -rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

# Main function to call:
def augment_imgs_of_case(channels, gt_lbls, roi_mask, wmaps_per_cat, prms, patch_size, rng=None, transf_mtx=None):
    '''If I get a rng, I would assume I want to do test-time augmentation'''
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # By Zeju: now it uses map_coordinate, similar to nnunet.
    # I dont know why, but previous implementation is sub-optimal
    # gt_lbls: np array of shape [x,y,z]. Can be None.
    # roi_mask: np array of shape [x,y,z]. Can be None.
    # wmaps_per_cat: List of np.arrays (floats or ints), weightmaps for sampling. Can be None.
    # prms: None (for no augmentation) or Dictionary with parameters of each augmentation type. }

    if prms is not None:

        (channels,
         gt_lbls,
         roi_mask,
         wmaps_per_cat,
         transf_mtx) = random_affine_deformation(channels,
                                                    gt_lbls,
                                                    roi_mask,
                                                    wmaps_per_cat,
                                                    patch_size,
                                                    rng,
                                                    prms,
                                                    transf_mtx)
    return channels, gt_lbls, transf_mtx


def random_affine_deformation(channels, gt_lbls, roi_mask, wmaps_l, patch_size, rng, prms, transf_mtx):
    if prms is None:
        return channels, gt_lbls, roi_mask, wmaps_l

    augm = AugmenterAffine(prob=prms['prob'],
                           rot_xyz=prms['rot_xyz'],
                           scaling=prms['scaling'],
                           seed=prms['seed'])
    if transf_mtx is None:
        transf_mtx = augm.roll_dice_and_get_random_transformation(rng)
    assert transf_mtx is not None

    channels = augm(images_l=channels,
                    transf_mtx=transf_mtx,
                    interp_orders=prms['interp_order_imgs'],
                    boundary_modes=prms['boundary_mode'],
                    patch_size=patch_size,
                    imgchannels=True)
    if gt_lbls is not None:
        (gt_lbls,
        roi_mask) = augm(images_l=[gt_lbls, roi_mask],
                        transf_mtx=transf_mtx,
                        interp_orders=[prms['interp_order_lbls'], prms['interp_order_roi']],
                        boundary_modes=prms['boundary_mode'], patch_size=patch_size, imgchannels=False)
    wmaps_l = augm(images_l=wmaps_l,
                   transf_mtx=transf_mtx,
                   interp_orders=prms['interp_order_wmaps'],
                   boundary_modes=prms['boundary_mode'], patch_size=patch_size, imgchannels=True)

    return channels, gt_lbls, roi_mask, wmaps_l, transf_mtx


class AugmenterParams(object):
    # Parent class, for parameters of augmenters.
    def __init__(self, prms):
        # prms: dictionary
        self._prms = collections.OrderedDict()
        self._set_from_dict(prms)

    def __str__(self):
        return str(self._prms)

    def __getitem__(self, key):  # overriding the [] operator.
        # key: string.
        return self._prms[key] if key in self._prms else None

    def __setitem__(self, key, item):  # For instance[key] = item assignment
        self._prms[key] = item

    def _set_from_dict(self, prms):
        if prms is not None:
            for key in prms.keys():
                self._prms[key] = prms[key]


class AugmenterAffineParams(AugmenterParams):
    def __init__(self, prms):
        # Default values.
        self._prms = collections.OrderedDict([('prob', 0.0),
                                              ('rot_xyz', (45., 45., 45.)),
                                              ('scaling', .1),
                                              ('seed', None),
                                              # For calls.
                                              ('interp_order_imgs', 3),
                                              ('interp_order_lbls', 1),
                                              ('interp_order_roi', 0),
                                              ('interp_order_wmaps', 1),
                                            #   ('boundary_mode', 'nearest'),
                                              ('boundary_mode', 'constant'),
                                              ('cval', 0.)])
        # Overwrite defaults with given.
        self._set_from_dict(prms)

    def __str__(self):
        return str(self._prms)


class AugmenterAffine(object):
    def __init__(self, prob, rot_xyz, scaling, seed=None):
        self.prob = prob  # Probability of applying the transformation.
        self.rot_xyz = rot_xyz
        self.scaling = scaling
        self.rng = np.random.RandomState(seed)

    def roll_dice_and_get_random_transformation(self, rng):
        if self.rng.random_sample() > self.prob:
            return -1  # No augmentation
        else:
            return self._get_random_transformation(rng)  # transformation for augmentation

    def _get_random_transformation(self, rng):
        local_state = np.random.RandomState()

        if rng == None:
            ## training mode
            rng1 = local_state.choice((1, -1))
            rng2 = local_state.choice((1, -1))
            rng3 = local_state.choice((1, -1))
            rng4 = local_state.choice((1, -1))
            '''if it could be divided by 10, make it have range (N-10, N)'''
            '''However, if it is minus, I would assume it comes from DM, I just choose from a uniform distribution.'''
            if self.rot_xyz[0] == 0:
                theta_x = self.rot_xyz[0] * np.pi / 180.
            elif self.rot_xyz[0] > 0:
                ## this is what I want to get
                ## just make sure I do not do anything wrong...
                theta_x = rng1 * local_state.uniform(np.max((0, self.rot_xyz[0] - 5.)), self.rot_xyz[0] + 5.) * np.pi / 180.
            else:
                theta_x = rng1 * local_state.uniform(0, - self.rot_xyz[0]) * np.pi / 180.

            if self.rot_xyz[1] == 0:
                theta_y = self.rot_xyz[1] * np.pi / 180.
            elif self.rot_xyz[1] > 0:
                ## this is what I want to get
                ## just make sure I do not do anything wrong...
                theta_y = rng2 * local_state.uniform(np.max((0, self.rot_xyz[1] - 5.)), self.rot_xyz[1] + 5.) * np.pi / 180.
            else:
                theta_y = rng2 * local_state.uniform(0, - self.rot_xyz[1]) * np.pi / 180.
            
            if self.rot_xyz[2] == 0:
                theta_z = self.rot_xyz[2] * np.pi / 180.
            elif self.rot_xyz[2] > 0:
                ## this is what I want to get
                ## just make sure I do not do anything wrong...
                theta_z = rng3 * local_state.uniform(np.max((0, self.rot_xyz[2] - 5.)), self.rot_xyz[2] + 5.) * np.pi / 180.
            else:
                theta_z = rng3 * local_state.uniform(0, - self.rot_xyz[2]) * np.pi / 180.

            if self.scaling > 0:
                scalingfactor = 1 + local_state.uniform(np.max((0, self.scaling - 0.05)), self.scaling + 0.05)
            else:
                scalingfactor = 1 + local_state.uniform(0, - self.scaling)

            if rng4 < 1:
                scalingfactor = 1 / scalingfactor
            scale = np.eye(3, 3) * scalingfactor
        else:
            rng1 = rng
            rng2 = rng
            rng3 = rng
            rng4 = rng
            theta_x = rng1 * self.rot_xyz[0] * np.pi / 180.
            theta_y = rng2 * self.rot_xyz[1] * np.pi / 180.
            theta_z = rng3 * self.rot_xyz[2] * np.pi / 180.
            scalingfactor = 1 + self.scaling
            if rng4 < 1:
                scalingfactor = 1 / scalingfactor
            scale = np.eye(3, 3) * scalingfactor

        rot_x = np.array([[np.cos(theta_x), -np.sin(theta_x), 0.],
                          [np.sin(theta_x), np.cos(theta_x), 0.],
                          [0., 0., 1.]])

        rot_y = np.array([[np.cos(theta_y), 0., np.sin(theta_y)],
                          [0., 1., 0.],
                          [-np.sin(theta_y), 0., np.cos(theta_y)]])

        rot_z = np.array([[1., 0., 0.],
                          [0., np.cos(theta_z), -np.sin(theta_z)],
                          [0., np.sin(theta_z), np.cos(theta_z)]])

        # Sample the scale (zoom in/out)
        # TODO: Non isotropic?
        # Affine transformation matrix.
        transformation_mtx = np.dot(scale, np.dot(rot_z, np.dot(rot_x, rot_y)))

        return transformation_mtx

    def _apply_transformation(self, image, coords, interp_order=2., boundary_mode='nearest', cval=0., imgchannels=False):
        # image should be 3 dimensional (Height, Width, Depth). Not multi-channel.
        # interp_order: Integer. 1,2,3 for images, 0 for nearest neighbour on masks (GT & brainmasks)
        # boundary_mode = 'constant', 'min', 'nearest', 'mirror...
        # cval: float. value given to boundaries if mode is constant.
        assert interp_order in [0, 1, 2, 3]

        mode = boundary_mode
        if mode == 'min':
            cval = np.min(image)
            mode = 'constant'

        # For recentering
        '''be aware that it should be different for even and odd'''
        '''for the default setting, shape is even, so nnunet would not worry too much'''
        '''but for DM, it is odd, and I should pick the floor'''
        for d in range(len(image.shape)):
            ctr = int(np.floor(image.shape[d] / 2.))
            coords[d] += ctr

        if imgchannels==False and interp_order != 0:
            unique_labels = np.unique(image)
            result = np.zeros(coords.shape[1:], image.dtype)
            for i, c in enumerate(unique_labels):
                new_image = scipy.ndimage.map_coordinates((image == c).astype(float),
                                                    coords,
                                                    order=interp_order,
                                                    mode=mode,
                                                    cval=cval)
                result[new_image >= 0.5] = c
            return result
        else:
            new_image = scipy.ndimage.map_coordinates(image.astype(float),
                                                    coords,
                                                    order=interp_order,
                                                    mode=mode,
                                                    cval=cval).astype(image.dtype)
            return new_image

    def __call__(self, images_l, transf_mtx, interp_orders, boundary_modes, patch_size, cval=0., imgchannels=False):
        # images_l : List of images, or an array where first dimension is over images (eg channels).
        #            An image (element of the var) can be None, and it will be returned unchanged.
        #            If images_l is None, then returns None.
        # transf_mtx: Given (from get_random_transformation), -1, or None.
        #             If -1, no augmentation/transformation will be done.
        #             If None, new random will be made.
        # intrp_orders : Int or List of integers. Orders of bsplines for interpolation, one per image in images_l.
        #                Suggested: 3 for images. 1 is like linear. 0 for masks/labels, like NN.
        # boundary_mode = String or list of strings. 'constant', 'min', 'nearest', 'mirror...
        # cval: single float value. Value given to boundaries if mode is 'constant'.
        if images_l is None:
            return None
        if transf_mtx is None:  # Get random transformation.
            transf_mtx = self.roll_dice_and_get_random_transformation()
        if not isinstance(transf_mtx, np.ndarray) and transf_mtx == -1:  # Do not augment
            return images_l
        # If scalars/string was given, change it to list of scalars/strings, per image.
        if isinstance(interp_orders, int):
            interp_orders = [interp_orders] * len(images_l[0])
        if isinstance(boundary_modes, str):
            boundary_modes = [boundary_modes] * len(images_l[0])
        
        '''I should be careful here'''
        ## For Unet, the patch size is like 80 * 80 * 80
        ## Therefore I should have this to make the rotation center is 0.5, and in the middle, neat!
        coords = create_zero_centered_coordinate_mesh(patch_size)

        coords = np.dot(coords.reshape(len(coords), -1).transpose(), transf_mtx).transpose().reshape(coords.shape)

        # Deform images.
        if type(images_l) is list:
            new_images = images_l
        else:
            new_images = np.zeros((images_l.shape[0], *patch_size))
        for img_i, int_order, b_mode in zip(range(len(images_l)), interp_orders, boundary_modes):
            if images_l[img_i] is None:
                pass  # Dont do anything. Let it be None.
            else:
                new_images[img_i] = self._apply_transformation(images_l[img_i],
                                                                coords.copy(),
                                                                int_order,
                                                                b_mode,
                                                                cval,
                                                                imgchannels)
        return new_images


############# Currently not used ####################

# DON'T use on patches. Only on images. Cause I ll need to find min and max intensities, to move to range [0,1]
def random_gamma_correction(channels, gamma_std=0.05):
    # Gamma correction: I' = I^gamma
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # IMPORTANT: Does not work if intensities go to negatives.
    if gamma_std is None or gamma_std == 0.:
        return channels

    n_channs = channels[0].shape[0]
    gamma = np.random.normal(1, gamma_std, [n_channs, 1, 1, 1])
    for path_idx in range(len(channels)):
        assert np.min(channels[path_idx]) >= 0.
        channels[path_idx] = np.power(channels[path_idx], 1.5, dtype='float32')

    return channels

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

def augment_sample(channels, gt_lbls, prms, Imgenlarge, rng=None, proof=0):
    '''If I get a rng, I would assume I want to do test-time augmentation'''
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # gt_lbls: np array of shape [x,y,z]
    # prms: None or Dictionary, with parameters of each augmentation type. }

    # it might have troubles from the multi-process. might be not, because every choice is different.
    # but I also create a local state here, just to be save
    local_state = np.random.RandomState()

    if prms is not None:
        # choose one augmentation here. the chosen one is set to prmssel
        channels, gt_lbls = random_rotation_90(channels, gt_lbls, prms['rotate90'], local_state, rng)
        
        # four color transformations
        # the order should not affect much, I follow the order of batchgenerator
        # the first three should access the whole image for some global statstics
        channels = random_guassian_noise(channels, prms['noise'], local_state, rng, proof)
        channels, Imgenlarge = random_guassian_blur(channels, prms['blur'], Imgenlarge.copy(), local_state, rng, proof)
        channels, Imgenlarge = random_histogram_distortion(channels, prms['hist_dist'], Imgenlarge, local_state, rng, proof)
        channels, Imgenlarge = random_contrast(channels, prms['contrast'], Imgenlarge, local_state, rng, proof)
        channels = simulate_low_resolution(channels, prms['simulowres'], local_state, rng, proof)

        channels, Imgenlarge = random_invgamma_correction(channels, prms['gamma'], Imgenlarge, local_state, rng, proof)
        channels, Imgenlarge = random_gamma_correction(channels, prms['gamma'], Imgenlarge, local_state, rng, proof)

        channels, gt_lbls = random_flip(channels, gt_lbls, prms['reflect'], local_state)

    return channels, gt_lbls


def random_histogram_distortion(channels, prms, Imgenlarge, local_state, rng, proof):
    # Shift and scale the histogram of each channel.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # prms: { 'shift': {'mu': 0.0, 'std':0.}, 'scale':{'mu': 1.0, 'std': '0.'} }
    if prms is None or prms['shift']['mu'] == 0 and prms['shift']['std'] == 0 and prms['scale']['mu'] ==0 and prms['scale']['std'] == 0:
        return channels, Imgenlarge

    if rng == None:
        ## training mode
        rng1 = local_state.choice((1, -1))
        rng2 = local_state.choice((1, -1))
        if prms['shift']['mu'] > 0:
            shiftmu = rng1 * local_state.uniform(np.max((0, prms['shift']['mu'] - 0.05)), prms['shift']['mu'] + 0.05)
            scalemu = 1 + local_state.uniform(np.max((0, prms['scale']['mu'] - 0.05)), prms['scale']['mu'] + 0.05)
        else:
            shiftmu = rng1 * local_state.uniform(0, - prms['shift']['mu'])
            scalemu = 1 + local_state.uniform(0, - prms['scale']['mu'])
    else:
        rng1 = rng
        rng2 = rng
        shiftmu = rng1 * prms['shift']['mu']
        scalemu = 1 + prms['scale']['mu']
    if rng2 < 0:
        scalemu = 1 / scalemu

    n_channs = channels[0].shape[0]
    if prms['shift'] is None:
        shift_per_chan = 0.
    elif prms['shift']['std'] != 0:  # np.random.normal does not work for an std==0.
        shift_per_chan = local_state.normal(shiftmu, prms['shift']['std'], [n_channs, 1, 1, 1])
    else:
        shift_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * shiftmu

    if prms['scale'] is None:
        scale_per_chan = 1.
    elif prms['scale']['std'] != 0:
        scale_per_chan = local_state.normal(scalemu, prms['scale']['std'], [n_channs, 1, 1, 1])
    else:
        scale_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * scalemu

    Imgenlarge = (Imgenlarge + shift_per_chan) * scale_per_chan

    # Intensity augmentation
    for path_idx in range(len(channels)):
        if proof == 0:
            channels[path_idx] = (channels[path_idx] + shift_per_chan) * scale_per_chan
        else:
            if np.sum(shift_per_chan) != 0 or np.mean(scale_per_chan) != 1:
                channels[path_idx] = channels[path_idx] * 0 - 1

    return channels, Imgenlarge

def random_contrast(channels, prms, Imgenlarge, local_state, rng, proof):
    # - mean and multiply a scalar
    # I should take the whole image as ref.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # prms: { 'factor': 0. }
    if prms is None or prms['factor'] == 0:
        return channels, Imgenlarge

    if rng == None:
        ## training mode
        rng = local_state.choice((1, -1))
        if prms['factor'] > 0:
            factor = 1 + local_state.uniform(np.max((0, prms['factor'] - 0.05)), prms['factor'] + 0.05)
        else:
            factor = 1 + local_state.uniform(0, prms['factor'])
    else:
        factor = 1 + prms['factor']
    
    if rng < 0:
        factor = 1 / factor

    mns = []
    maxms = []
    minms = []
    # in case we are sampling like Deepmedic, I should keep the pixel absolute intensity similar.
    for c in range(channels[0].shape[0]):
        mns.append(Imgenlarge[c].mean())
        maxms.append(Imgenlarge[c].max())
        minms.append(Imgenlarge[c].min())
        Imgenlarge[c] = (Imgenlarge[c] - mns[c]) * factor + mns[c]
        Imgenlarge[c][Imgenlarge[c] < minms[c]] = minms[c]
        Imgenlarge[c][Imgenlarge[c] > maxms[c]] = maxms[c]

    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            ## to retain stats
            channels[path_idx][c] = (channels[path_idx][c] - mns[c]) * factor + mns[c]
            channels[path_idx][c][channels[path_idx][c] < minms[c]] = minms[c]
            channels[path_idx][c][channels[path_idx][c] > maxms[c]] = maxms[c]

    return channels, Imgenlarge

def random_guassian_noise(channels, prms, local_state, rng, proof):
    # Add gaussian noise
    if prms is None or prms['std'] == 0:
        return channels

    if rng == None:
        ## training mode
        if prms['std'] > 0:
            noise_std = local_state.uniform(np.max((0, prms['std'] - 0.025)), prms['std'] + 0.025)
        else:
            noise_std = local_state.uniform(0, - prms['std'])
    else:
        noise_std = prms['std'] 

    # Intensity augmentation

    for path_idx in range(len(channels)):
        if proof == 0:
            shift_per_chan = local_state.normal(0, noise_std, [channels[path_idx].shape[0], channels[path_idx].shape[1],
                                                               channels[path_idx].shape[2], channels[path_idx].shape[3]])
            channels[path_idx] = channels[path_idx] + shift_per_chan
        else :
            # the ridiculous augmentation.
            if prms['std'] > 0 :
                # print('dangerous here')
                channels[path_idx] = channels[path_idx] * 0 + 1

    return channels

def random_guassian_blur(channels, prms, Imgenlarge, local_state, rng, proof):
    # Add gaussian noise
    if prms is None or prms['sigma'] == 0:
        return channels, Imgenlarge
    
    if rng == None:
        ## training mode
        if prms['sharpen'] == False:
            if prms['sigma'] > 0:
                blur_sigma = local_state.uniform(prms['sigma'] - 0.1, prms['sigma'] + 0.1)
            else:
                blur_sigma = local_state.uniform(0.4, - prms['sigma'])
        else:
            if prms['sigma'] > 0:
                blur_sigma = local_state.uniform(prms['sigma'] - 0.1, prms['sigma'] + 0.1)
            else:
                blur_sigma = local_state.uniform(- prms['sigma'], 1.)
    else:
        blur_sigma = prms['sigma']

    # save the statistic
    maxms = []
    minms = []
    for c in range(channels[0].shape[0]):
        maxms.append(Imgenlarge[c].max())
        minms.append(Imgenlarge[c].min())

    # blur
    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            if prms['sharpen'] == False:
                channels[path_idx][c] = gaussian_filter(channels[path_idx][c], blur_sigma, order=0)
            else:
                channels[path_idx][c] = channels[path_idx][c] - gaussian_laplace(channels[path_idx][c], blur_sigma)
                channels[path_idx][c][channels[path_idx][c] < minms[c]] = minms[c]
                channels[path_idx][c][channels[path_idx][c] > maxms[c]] = maxms[c]
                

    return channels, Imgenlarge

def simulate_low_resolution(channels, prms, local_state, rng, proof):
    # simulate the low resolution
    order_downsample=1
    order_upsample=0
    if prms is None or prms['zoom'] == 1:
        return channels

    if rng == None:
        ## training mode
        if prms['zoom'] > 0:
            simlscale = local_state.uniform(prms['zoom'] - 0.1, prms['zoom'] + 0.1)
        else:
            simlscale = local_state.uniform(- prms['zoom'], 1.)
    else:
        simlscale = prms['zoom']

    # zoom in
    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            shp = np.array(channels[path_idx].shape[1:])
            target_shape = np.round(shp * simlscale).astype(int)
            downsampled = resize(channels[path_idx][c].astype(float), target_shape, order=order_downsample, mode='edge',
                                anti_aliasing=False)
            channels[path_idx][c] = resize(downsampled, shp, order=order_upsample, mode='edge',
                                    anti_aliasing=False)

    return channels

def random_gamma_correction(channels, prms, Imgenlarge, local_state, rng, proof):
    # Gamma correction
    if prms is None or prms['gamma'] == 0:
        return channels, Imgenlarge

    if proof == 0:
        epsilon = 1e-6
        if rng == None:
            ## training mode
            rng = local_state.choice((1, -1))
            if prms['gamma'] > 0 :
                gamma = 1 + local_state.uniform(np.max((0, prms['gamma'] - 0.1)), prms['gamma'] + 0.1)
            else:
                gamma = 1 + local_state.uniform(0, - prms['gamma'])
        else:
            gamma = 1 + prms['gamma']
            # 
        if rng < 0:
            gamma = 1 / gamma
        
        minms = []
        rnges = []
        mns = []
        sds = []
        # in case we are sampling like Deepmedic, I should keep the pixel absolute intensity similar.
        for c in range(channels[0].shape[0]):
            minms.append(Imgenlarge[c].min())
            rnges.append(Imgenlarge[c].max() - Imgenlarge[c].min())
            mns.append(Imgenlarge[c].mean())
            sds.append(Imgenlarge[c].std())

        for path_idx in range(len(channels)):
            for c in range(channels[path_idx].shape[0]):
                ## to retain stats
                if rnges[c] != 0 and sds[c] != 0:
                    # Jan 27, 2021, fix a bug here.
                    minm = np.min((channels[path_idx][c].min(), Imgenlarge[c].min()))
                    maxm = np.max((channels[path_idx][c].max(), Imgenlarge[c].max()))
                    rnge = maxm - minm
                    # in case the minimum is sampled out, it should not happen very often I suppose
                    channels[path_idx][c] = np.power(((channels[path_idx][c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                    
                    mn = Imgenlarge[c].mean()
                    sd = Imgenlarge[c].std()

                    Imgenlarge[c] = np.power(((Imgenlarge[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm

                    mnafter = Imgenlarge[c].mean()
                    
                    Imgenlarge[c] = Imgenlarge[c] - mnafter + mn
                    sdafter = Imgenlarge[c].std()

                    Imgenlarge[c] = Imgenlarge[c] / (sdafter + epsilon) * (sd + epsilon)

                    channels[path_idx][c] = channels[path_idx][c] - mnafter + mn
                    channels[path_idx][c] = channels[path_idx][c] / (sdafter + epsilon) * (sd + epsilon)

                else:
                    # if it is a blank, do not process.
                    # it is different from nnunet, it does not happen this case.
                    channels[path_idx][c] = channels[path_idx][c]
    else:
        # the ridiculous augmentation.
        for path_idx in range(len(channels)):
            if prms['gamma'] > 0 :
                channels[path_idx] = - channels[path_idx] * 0

    return channels, Imgenlarge

def random_invgamma_correction(channels, prms, Imgenlarge, local_state, rng, proof):
    # Inverted Gamma correction
    if prms is None or prms['invgamma'] == 0:
        return channels, Imgenlarge

    epsilon = 1e-6
    if rng == None:
        ## training mode
        rng = local_state.choice((1, -1))
        if prms['invgamma'] > 0 :
            gamma = 1 + local_state.uniform(np.max((0, prms['invgamma'] - 0.1)), prms['invgamma'] + 0.1)
        else:
            gamma = 1 + local_state.uniform(0, - prms['invgamma'])
    else:
        gamma = 1 + prms['invgamma']

    if rng < 0:
        gamma = 1 / gamma
    
    minms = []
    rnges = []
    mns = []
    sds = []
    Imgenlarge = - Imgenlarge
    for c in range(channels[0].shape[0]):
        minms.append(Imgenlarge[c].min())
        rnges.append(Imgenlarge[c].max() - Imgenlarge[c].min())
        mns.append(Imgenlarge[c].mean())
        sds.append(Imgenlarge[c].std())

    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            channels[path_idx][c] = - channels[path_idx][c]
            if rnges[c] != 0 and sds[c] != 0:
                # Jan 27, 2021, fix a bug here.
                minm = np.min((channels[path_idx][c].min(), Imgenlarge[c].min()))
                maxm = np.max((channels[path_idx][c].max(), Imgenlarge[c].max()))
                rnge = maxm - minm
                # in case the minimum is sampled out, it should not happen very often I suppose
                channels[path_idx][c] = np.power(((channels[path_idx][c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                
                mn = Imgenlarge[c].mean()
                sd = Imgenlarge[c].std()

                Imgenlarge[c] = np.power(((Imgenlarge[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm

                mnafter = Imgenlarge[c].mean()
                
                Imgenlarge[c] = Imgenlarge[c] - mnafter + mn
                sdafter = Imgenlarge[c].std()

                Imgenlarge[c] = Imgenlarge[c] / (sdafter + epsilon) * (sd + epsilon)

                channels[path_idx][c] = channels[path_idx][c] - mnafter + mn
                channels[path_idx][c] = channels[path_idx][c] / (sdafter + epsilon) * (sd + epsilon)

            else:
                # if it is a blank, do not process.
                # it is different from nnunet, it does not happen this case.
                channels[path_idx][c] = channels[path_idx][c]
            channels[path_idx][c] = - channels[path_idx][c]

    return channels, - Imgenlarge

def random_flip(channels, gt_lbls, probs_flip_axes, local_state):
    # Flip (reflect) along each axis.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # gt_lbls: np array of shape [x,y,z]
    # probs_flip_axes: list of probabilities, one per axis.
    if probs_flip_axes is None:
        return channels, gt_lbls

    for axis_idx in range(len(channels[0].shape[1:])):  # 3 dims
        flip = local_state.choice(a=(True, False), size=1, p=(probs_flip_axes[axis_idx], 1. - probs_flip_axes[axis_idx]))
        if flip:
            for path_idx in range(len(channels)):
                channels[path_idx] = np.flip(channels[path_idx], axis=axis_idx + 1)  # + 1 because dim [0] is channels.
            if gt_lbls is not None:
                gt_lbls = np.flip(gt_lbls, axis=axis_idx)

    return channels, gt_lbls


def random_rotation_90(channels, gt_lbls, probs_rot_90, local_state, rng):
    # Rotate by 0/90/180/270 degrees.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # gt_lbls: np array of shape [x,y,z]
    # probs_rot_90: {'xy': {'0': fl, '90': fl, '180': fl, '270': fl},
    #                'yz': {'0': fl, '90': fl, '180': fl, '270': fl},
    #                'xz': {'0': fl, '90': fl, '180': fl, '270': fl} }
    if probs_rot_90 is None:
        return channels, gt_lbls

    if rng == None:
        rng = local_state.choice((1, -1))

    for key, plane_axes in zip(['xy', 'yz', 'xz'], [(0, 1), (1, 2), (0, 2)]):
        probs_plane = probs_rot_90[key]

        if probs_plane is None:
            continue

        assert len(probs_plane) == 4  # rotation 0, rotation 90 degrees, 180, 270.
        # assert channels[0].shape[1 + plane_axes[0]] == channels[0].shape[1 + plane_axes[1]]  
        # # +1 cause [0] is channel. Image/patch must be isotropic.

        # Normalize probs
        sum_p = probs_plane['0'] + probs_plane['90'] + probs_plane['180'] + probs_plane['270']
        if sum_p == 0:
            continue
        for rot_k in probs_plane:
            probs_plane[rot_k] /= sum_p  # normalize p to 1.

        p_rot_90_x0123 = (probs_plane['0'], probs_plane['90'], probs_plane['180'], probs_plane['270'])

        if np.max(p_rot_90_x0123) < 1:
            # need rng to make the choice]
            if key == 'xz':
                ## this direction is inverse
                if rng == -1:
                    p_rot_90_x0123 = (probs_plane['0'], 1.0, probs_plane['180'], 0.0)
                else:
                    p_rot_90_x0123 = (probs_plane['0'], 0.0, probs_plane['180'], 1.0)
            else:    
                if rng == 1:
                    p_rot_90_x0123 = (probs_plane['0'], 1.0, probs_plane['180'], 0.0)
                else:
                    p_rot_90_x0123 = (probs_plane['0'], 0.0, probs_plane['180'], 1.0)

        rot_90_xtimes = local_state.choice(a=(0, 1, 2, 3), size=1, p=p_rot_90_x0123)

        for path_idx in range(len(channels)):
            channels[path_idx] = np.rot90(channels[path_idx], k=rot_90_xtimes,
                                          axes=[axis + 1 for axis in plane_axes])  # + 1 cause [0] is channels.
        if gt_lbls is not None:
            gt_lbls = np.rot90(gt_lbls, k=rot_90_xtimes, axes=plane_axes)

    return channels, gt_lbls
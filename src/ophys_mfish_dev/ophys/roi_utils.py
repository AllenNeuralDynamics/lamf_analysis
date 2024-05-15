import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from skimage import measure
from typing import Union, Tuple
import cv2


###################################################################################################
# I/O
###################################################################################################


def napari_to_rois_list(path):
    """Load a napari labeled image into a numpy array
    """
    img = tif.imread(path)

    # take each unique value and make it a binary mask, collect all masks in a list
    roi_masks = []
    for i in np.unique(img):
        mask = np.zeros(img.shape)
        mask[img == i] = 1
        roi_masks.append(mask)

    return roi_masks


def load_napari_tif(path):
    """Load a napari labeled image into a numpy array"""
    img = tif.imread(path)

    return img


def s2p_rois_to_mask(stat, output_shape=None):
    """Take a stat file from suite2p and convert it to list of roi masks"""
    
    if output_shape is None:
        output_shape = (512, 512)
        print(f"output_shape for roi canvas not specified, using {output_shape}")
    roi_masks = []
    for i in range(len(stat)):
        mask = np.zeros(output_shape)
        soma_crop = stat[i]["soma_crop"]

        # use soma crop to create mask
        ypix = stat[i]["ypix"]
        xpix = stat[i]["xpix"]

        mask[ypix[soma_crop], xpix[soma_crop]] = 1
        roi_masks.append(mask)

    return roi_masks

###################################################################################################
# metrics
###################################################################################################


def get_roi_center(mask):
    """Get the x, y center of mass of roi mask"""
    ypix, xpix = np.nonzero(mask)
    x_center = np.mean(xpix)
    y_center = np.mean(ypix)

    return x_center, y_center


def calc_pixel_accuracy(roi_gt, roi_pred):
    """

    Parameters
    ----------
    roi_gt : numpy array
        ground truth roi image

    roi_pred : numpy array
        predicted roi image
    """
    roi_labeled_b = roi_gt > 0
    roi_pred_b = roi_pred > 0
    pixel_accuracy = np.sum(roi_labeled_b == roi_pred_b) / np.prod(roi_labeled_b.shape)
    return pixel_accuracy


def calc_iou(roi_gt, roi_pred):
    """
    Calculate intersection over union of two roi masks
    """
    roi_labeled_b = roi_gt > 0
    roi_pred_b = roi_pred > 0

    intersection = np.sum(roi_labeled_b * roi_pred_b)
    union = np.sum(roi_labeled_b) + np.sum(roi_pred_b) - intersection
    iou = intersection / union

    return iou


def roi_bounding_box(roi_mask, pad: int = None):
    """Returns the bounding box of a roi mask"""
    roi_mask = roi_mask.astype(np.uint8)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)

    if pad:
        x = x - pad
        y = y - pad
        w = w + pad * 2
        h = h + pad * 2
    return x, y, w, h


###################################################################################################
# Working with ROI masks
###################################################################################################

def reorder_mask_im(mask_im):
    ''' Reorder ROI IDs in the mask image to be consecutive starting from 1
    mask_im: input mask. Must be 2 dimensional    
    '''
    assert len(mask_im.shape) == 2
    if len(np.unique(mask_im)) - 1 != mask_im.max():
        old_ids = np.setdiff1d(mask_im.unique(), 0)
        new_mask_im = np.zeros(mask_im.shape)
        for i, old_id in enumerate(old_ids):
            id_pix = np.where(mask_im == old_id)
            new_mask_im[id_pix[0], id_pix[1]] = i
        return new_mask_im
    else:
        return mask_im
    

def make_3d_mask_from_2d(mask_im):
# make a 3d mask from 2d mask
    mask_im = reorder_mask_im(mask_im)
    mask_3d = np.zeros((mask_im.max(), *mask_im.shape))
    for i in range(mask_im.max()):
        pix = np.where(mask_im==(i+1))
        mask_3d[i, pix[0], pix[1]] = 1
    return mask_3d


def get_moved_mask_3d(moving_mask_3d,
                      sr, # StackReg object
                      thresholding: Union['num_pix', 'pix_threshold']='num_pix',
                      pix_threshold=0.5):                 
    moved_mask_3d = np.zeros(moving_mask_3d.shape)
    for i in range(moving_mask_3d.shape[0]):
        temp_moving_im = moving_mask_3d[i, :, :]
        temp_moved_im = sr.transform(temp_moving_im)
        if thresholding == 'pix_threshold':
            temp_moved_im_thresholded = temp_moved_im > pix_threshold
        elif thresholding == 'num_pix':
            num_pix = len(np.where(temp_moving_im.flatten())[0])
            moved_inds = np.unravel_index(np.argsort(-temp_moved_im.flatten())[:num_pix], temp_moving_im.shape)
            temp_moved_im_thresholded = np.zeros(temp_moved_im.shape)
            temp_moved_im_thresholded[moved_inds] = 1
        else:
            raise ValueError('thresholding must be either "num_pix" or "pix_threshold".')
        moved_mask_3d[i,:,:] = temp_moved_im_thresholded
    return moved_mask_3d


def get_2d_mask_from_3d(mask_3d):
    # deal with the overlap?
    # for now, assign higher index value
    mask_2d = np.argmax(mask_3d, axis=0)

    return mask_2d


###############
# ROI collation
# TODO: needs test

def collate_masks(mask_3d1, mask_3d2, iou_threshold=0.3):
    """Collate two masks into one.
    Plus, calculate unique ROIs going forward and backward (from mask_3d1 to mask_3d2 and vice versa)
    using IoU threshold
    """
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    if mask_3d1.shape[0] == 0:
        if mask_3d2.shape[0] == 0:
            return np.zeros((0, mask_3d1.shape[1], mask_3d1.shape[2]), dtype=np.uint16), (0,0)
        else:
            return mask_3d2, (mask_3d2.shape[0], 0)
    elif mask_3d2.shape[0] == 0:
        return mask_3d1, (0, mask_3d1.shape[0])
    else:
        assert mask_3d1.max() == mask_3d1.shape[0]
        assert mask_3d2.max() == mask_3d2.shape[0]
    iou_mat, ioa_mat1, ioa_mat2 = get_iou_ioa_mat(mask_3d1, mask_3d2)

    num_unique_in_mask1 = len(np.where(np.sum(iou_mat >= iou_threshold, axis=1) == 0)[0])
    num_unique_in_mask2 = len(np.where(np.sum(iou_mat >= iou_threshold, axis=0) == 0)[0])

    ind_remove_from_mask1, ind_remove_from_mask2, \
        ioa_mat1, ioa_mat2 = deal_with_multi_overlap(ioa_mat1, ioa_mat2,
                                                    mask_3d1, mask_3d2)
    ind_remove_from_mask1, ind_remove_from_mask2 \
        = deal_with_one_to_one(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2,
                               ind_remove_from_mask1, ind_remove_from_mask2)
    new_mask = clean_and_merge_masks(mask_3d1, mask_3d2, 
                                     ind_remove_from_mask1,
                                     ind_remove_from_mask2)
    return new_mask, (num_unique_in_mask1, num_unique_in_mask2), (ind_remove_from_mask1, ind_remove_from_mask2)


def get_iou_ioa_mat(mask_3d1, mask_3d2):
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[0]
    assert mask_3d2.max() == mask_3d2.shape[0]

    iou_mat = np.zeros((mask_3d1.shape[0], mask_3d2.shape[0])) # intersection / union
    ioa_mat1 = np.zeros((mask_3d1.shape[0], mask_3d2.shape[0])) # intersection / area of mask1 ROI
    ioa_mat2 = np.zeros((mask_3d1.shape[0], mask_3d2.shape[0])) # intersection / area of mask2 ROI

    for i in range(mask_3d1.shape[0]):
        for j in range(mask_3d2.shape[0]):
            intersection_map = mask_3d1[i] * mask_3d2[j] > 0
            if intersection_map.any():
                union_map = mask_3d1[i] + mask_3d2[j] > 0
                iou_mat[i,j] = np.sum(intersection_map) / np.sum(union_map)
                ioa_mat1[i,j] = np.sum(intersection_map) / np.sum(mask_3d1)
                ioa_mat2[i,j] = np.sum(intersection_map) / np.sum(mask_3d2)
    #TODO: needs validation in edge cases?
    return iou_mat, ioa_mat1, ioa_mat2


def deal_with_multi_overlap(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2, ioa_threshold=0.5):
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # Only care about multi-to-one, since that was the only cases that I've seen.
    # When there are multiple rois from mask2 that overlap with one roi from mask1
    multi_ind1 = np.where(np.sum(ioa_mat2 >= ioa_threshold, axis=1)>1)[0]
    # When there are multiple rois from mask1 that overlap with one roi from mask2
    multi_ind2 = np.where(np.sum(ioa_mat1 >= ioa_threshold, axis=0)>1)[0]
    # These two are assumed to be disjoint, because there was no multi-to-multi case

    area_dist1 = np.array([np.sum(mask_3d1[i]) for i in range(mask_3d1.shape[0])])
    area_dist2 = np.array([np.sum(mask_3d2[i]) for i in range(mask_3d2.shape[0])])
    area_dist = np.concatenate([area_dist1, area_dist2])

    ind_remove_from_mask1 = []
    ind_remove_from_mask2 = []

    for ind in multi_ind1:
        temp_overlap_roi_inds2 = np.where(ioa_mat2[ind,:] >= ioa_threshold)[0]
        # pick the one(s) with closer to median area
        percentile_to_med1 = np.abs(0.5 - np.sum(area_dist < area_dist1[ind])/len(area_dist))
        percentile_to_med2 = [np.abs(0.5 - np.sum(area_dist < area_dist2[i])/len(area_dist)) for i in temp_overlap_roi_inds2]
        if (percentile_to_med2 > percentile_to_med1).any():
            for temp_ind in temp_overlap_roi_inds2:
                ind_remove_from_mask2.append(temp_ind)
        else:
            ind_remove_from_mask1.append(ind)

    # repeat for multi_ind2
    for ind in multi_ind2:
        temp_overlap_roi_inds1 = np.where(ioa_mat1[:,ind] >= ioa_threshold)[0]
        # pick the one(s) with closer to median area
        percentile_to_med2 = np.abs(0.5 - np.sum(area_dist < area_dist2[ind])/len(area_dist))
        percentile_to_med1 = [np.abs(0.5 - np.sum(area_dist < area_dist1[i])/len(area_dist)) for i in temp_overlap_roi_inds1]
        if (percentile_to_med1 > percentile_to_med2).any():
            for temp_ind in temp_overlap_roi_inds1:
                ind_remove_from_mask1.append(temp_ind)
        else:
            ind_remove_from_mask2.append(ind)
    for ind in ind_remove_from_mask1:
        ioa_mat1[ind,:] = 0
        ioa_mat2[ind,:] = 0
    for ind in ind_remove_from_mask2:
        ioa_mat1[:,ind] = 0
        ioa_mat2[:,ind] = 0
    return ind_remove_from_mask1, ind_remove_from_mask2, ioa_mat1, ioa_mat2


def deal_with_one_to_one(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2,
                        ind_remove_from_mask1, ind_remove_from_mask2,
                        ioa_threshold=0.5):
    ###
    # After multi_overlap is dealt with
    ###
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # Sweeping through one direction is enough, because only one-to-one match (if any) is left
    area_dist1 = np.array([np.sum(mask_3d1[i]) for i in range(mask_3d1.shape[0])])
    area_dist2 = np.array([np.sum(mask_3d2[i]) for i in range(mask_3d2.shape[0])])
    area_dist = np.concatenate([area_dist1, area_dist2])
    for ind1 in range(ioa_mat1.shape[0]):
        if (ioa_mat1[ind1,:] >= ioa_threshold).any() or (ioa_mat2[ind1,:] >= ioa_threshold).any():
            match_ind2 = np.concatenate([np.where(ioa_mat1[ind1,:] >= ioa_threshold)[0],
                                         np.where(ioa_mat2[ind1,:] >= ioa_threshold)[0]])
            match_ind2 = np.unique(match_ind2)

            percentile_to_med1 = np.abs(0.5 - np.sum(area_dist < area_dist1[ind1])/len(area_dist))
            percentile_to_med2 = np.abs(0.5 - np.sum(area_dist < np.sum(area_dist2[match_ind2]))/len(area_dist))
            # summing area_dist2 in case where there are multiple overlapping rois in mask2
            # It's rare, but when there is, it shows neighboring ROIs that seem like one ROI
            if percentile_to_med1 < percentile_to_med2:
                ind_remove_from_mask2.extend(match_ind2)
            else:
                ind_remove_from_mask1.append(ind1)
    return ind_remove_from_mask1, ind_remove_from_mask2


def clean_and_merge_masks(mask_3d1, mask_3d2, ind_remove_from_mask1, ind_remove_from_mask2):
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[0]
    assert mask_3d2.max() == mask_3d2.shape[0]
    mask_3d1 = np.delete(mask_3d1, ind_remove_from_mask1)
    mask_3d2 = np.delete(mask_3d2, ind_remove_from_mask2)
    mask_3d_merged = np.dstack([mask_3d1, mask_3d2])
    return mask_3d_merged



###################################################################################################
# Plotting
###################################################################################################


def plot_contours_overlap_two_masks(mask1: np.ndarray,
                                    mask2: np.ndarray,
                                    img: np.ndarray = None,
                                    colors: list = None,
                                    ax=None) -> plt.axes:
    """Given two masks, plot the contours of the masks on an image

    Parameters
    ----------
    mask1 : np.ndarray
        mask 1, can be 2D or 3D. If 3D, axis=0 means cell ind (2D shape by mask1.shape[-2:])
    mask2 : np.ndarray
        mask 2, same as in mask 1
    img : np.ndarray, optional
        background image
    colors : list, optional
        list of colors for each mask, by default None
    ax : plt.axes, optional
        axis to plot on

    Returns
    -------
    plt.axes
    """
    # assign mask shapes and ensure 3d shape and shape matching between the masks
    if len(mask1.shape)==2:
        mask1shape = mask1.shape
        mask2shape = mask2.shape
    elif len(mask1.shape)==3:
        mask1shape = mask1.shape[1:]
        mask2shape = mask2.shape[1:]
    else:
        raise ValueError('Mask should be 2D or 3D.')    
    assert mask1shape == mask2shape, "masks must be same shape in 2D"
    if len(mask1.shape)==2:
        mask1_3d = make_3d_mask_from_2d(mask1)
    else:
        mask1_3d = mask1
    if len(mask2.shape)==2:
        mask2_3d = make_3d_mask_from_2d(mask2)
    else:
        mask2_3d = mask2

    # Initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if colors is None:
        colors = ['r', 'b']

    # background image
    if img is not None:
        assert img.shape == mask1shape
        vmax = np.percentile(img, 99.6)
        ax.imshow(img, vmax=vmax, cmap=plt.cm.gray)
    else:
        print()
        bg = np.ones(mask1shape)
        ax.imshow(bg, cmap=plt.cm.gray, vmin=0, vmax=1)

    # Plot contours
    for i in range(mask1_3d.shape[0]):
        contour = measure.find_contours(mask1_3d[i, :, :], 0.5)
        ax.plot(contour[0][:, 1], contour[0][:, 0], linewidth=1, color=colors[0],
                alpha=0.6)
    for i in range(mask2_3d.shape[0]):
        contour = measure.find_contours(mask2_3d[i, :, :], 0.5)
        ax.plot(contour[0][:, 1], contour[0][:, 0], linewidth=1, color=colors[1],
                alpha=0.6)

    ax.set_facecolor('white')

    return ax


# TODO: typical plots for 
# def plot_rois_gt_vs_pred(mask1: np.ndarray,
#                         mask2: np.ndarray,
#                         img: np.ndarray = None,
#                         white_bg: bool = False,
#                         title: str = '',
#                         ax=None,
#                         save_dir: str = None,
#                         save: bool = False):
    
#     ax = plot_contours_overlap_two_masks(mask1, mask2, img, ax=ax, colors = [''])
        

    
#     ax.set_xticks([])
#     ax.set_yticks([])
#     if title:
#         ax.set_title(title)
#     else:
#         title = "red=gt_green_pred"
#         ax.set_title("red = gt, green = pred")
#     if save:
#         output_fn = save_dir + title + ".png"
#         plt.savefig(output_fn, bbox_inches='tight', dpi=300, facecolor='white', transparent=False)

###################################################################################################
# Ground truth v Prediction evalulation
###################################################################################################


def classify_single_pred_roi_by_iou(roi_mask_pred: np.ndarray,
                                    all_gt_masks: np.ndarray,
                                    tp_thresh: float = .3):
    """Calculate iou of single mask with all other masks from a predicted set

    roi_mask_pred: np.array
        mask of single roi from predicted image
    all_gt_masks: np.array
        array of all gt masks from single plane
    tp_thresh: float
        threshold for iou to be considered a true positive

    Returns
    max_iou: float
        max iou of roi_mask_pred with all other masks
    max_roi: int
        roi id of mask with max iou
    classification: str
        classification of roi based on max_iou

    """
    roi_ids_gt = np.unique(all_gt_masks)
    roi_id_pred = np.unique(roi_mask_pred)[0]

    all_ious_dict = {}

    ious = []
    # binarize roi_mask_pred
    roi_mask_pred = roi_mask_pred > 0
    for roi in roi_ids_gt:

        roi_pred_mask = all_gt_masks == roi

        # binarize roi_pred_mask (need?)
        roi_pred_mask = roi_pred_mask > 0

        # calc intersection
        intersection = np.sum(roi_mask_pred * roi_pred_mask)
        # calc union
        union = np.sum(roi_mask_pred) + np.sum(roi_pred_mask) - intersection
        # calc iou
        iou = intersection / union

        ious.append(iou)

    all_ious_dict.update({roi_id_pred: ious})

    # get max iou
    max_iou = np.round(np.max(ious), 3)
    max_roi = roi_ids_gt[np.argmax(ious)]

    # if multiple rois have max iou over thresh, then MP
    if np.sum(np.array(ious) > tp_thresh) > 1:
        classification = "MP"  # "Multiple Positive"
    elif max_iou >= tp_thresh:
        classification = "TP"  # "True Positive"
    elif max_iou < tp_thresh:
        classification = "FP"  # "False Negative"

    return max_iou, max_roi, classification, all_ious_dict


def calc_roi_classification_stats(roi_masks_pred, roi_masks_gt, tp_thresh = .3):
    roi_class_dict = {}
    all_ious_dict = {}

    roi_ids_gt = np.unique(roi_masks_gt)
    roi_ids_pred = np.unique(roi_masks_pred)

    for roi_id_pred in roi_ids_pred:
        roi_mask = roi_masks_pred == roi_id_pred
        max_iou, gt_roi_id, classification, ious_dict = classify_single_pred_roi_by_iou(roi_mask, 
                                                                                       roi_masks_gt,
                                                                                       tp_thresh = tp_thresh)

        roi_class_dict.update({roi_id_pred: {"iou": max_iou,
                               "gt_roi_id": gt_roi_id,
                               "classification": classification}})
        print(f"iou for roi_pred: {roi_id_pred} is {max_iou} with roi_gt: {gt_roi_id}")

        all_ious_dict.update({roi_id_pred: ious_dict})

    # convert to dataframe
    roi_class_df = pd.DataFrame.from_dict(roi_class_dict, orient="table")
    roi_class_df = roi_class_df.reset_index().rename(columns={"index": "pred_roi_id"})

    # add FN to new df
    fp_roi_ids = np.setdiff1d(roi_ids_gt, roi_class_df["gt_roi_id"].unique())
    fp_df = pd.DataFrame({"pred_roi_id": fp_roi_ids,
                          "classification": "FN"})
    roi_class_df_fn = pd.concat([roi_class_df, fp_df], ignore_index=True)

    return roi_class_df_fn



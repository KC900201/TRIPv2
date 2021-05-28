"""
@author: kcng
@filename: trip_dataset.py
@coding: utf-8
========================
Date          Comment
========================
05152021      First revision
"""

import os
import numpy as np
import math
import torch.utils.data as D

import albumentations as A

from albumentations.pytorch import ToTensorV2  # future usage
from functools import lru_cache


class TripDataset(D.Dataset):  # Replace chainer.dataset.DatasetMixin with torch.utils.data.Dataset
    """
        A class of TRIP(Traffic Risk Prediction) dataset
        Args:
            ds_path (str): a dataset path
            spec_file (str): a dataset spec file name
            layer_name (str): feature extraction layer used (33, 39, 45) for Yolov3 bounding box
            box_type (str): a type of boxes - 'tbox' or 'ebox' - (used only for feature type 'raw')
            sp (int): extract only wanted features with skipped interval
        Returns:
            h (a Variable of hidden state array): a hidden state array
    """

    @lru_cache()
    def __init__(self, ds_path, spec_file, layer_name, box_type=None, sp=0):
        layer_names = ''
        if (',' in layer_name):
            layer_names = str(layer_name).strip().split(',')
        else:
            layer_names = str(layer_name).strip()

        self.ds_path = ds_path
        os.chdir('C:/Users/atsumilab/Pictures/TRIP_Dataset')  # default dataset path

        with open(os.path.join(self.ds_path, spec_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('feature'):
                self.feature_type = line[line.find(':') + 1:].strip()
            elif line.startswith('layer'):
                layers = line[line.find(':') + 1:].split(';')
                for layer in layers:
                    quintuple = layer.strip().split(',')
                    quintuple = [element.strip() for element in quintuple]
                    if quintuple[0] in layer_names:
                        layer_dir = quintuple[1]
                        # Upscaling of height and width for layer 33 and 39 - 20190130
                        self.layer_info = (quintuple[0], int(quintuple[2]), int(quintuple[3]),
                                           int(quintuple[4]))  # (layer_name, height, width, channels)

        if self.feature_type == 'raw':
            self.box_type = box_type
        elif self.feature_type == 'tbox_processed':
            self.box_type = 'tbox'
        elif self.feature_type == 'ebox_processed':
            self.box_type = 'ebox'

        # set dataset (feature_data, box_data)
        # feature data is a list of each feature file sequence(list)
        # box data is a list of each box file sequence(list)

        self.dirs = [d for d in os.listdir(self.ds_path) if os.path.isdir(os.path.join(self.ds_path, d))]
        self.feature_data = []
        self.box_data = []

        for d in self.dirs:
            elist = []
            if (type(layer_names) == str):
                flist = [os.path.join(d, layer_names, f) for f in
                         os.listdir(os.path.join(self.ds_path, d, layer_names))]
                if (sp > 0):
                    flist = flist[sp:len(flist)]
                elist.append(flist)
            else:
                for layer in layer_names:
                    flist = [os.path.join(d, layer, f) for f in os.listdir(os.path.join(self.ds_path, d, layer))]
                    if (sp > 0):
                        flist = flist[sp:len(flist)]
                    elist.append(flist)
            self.feature_data.append(list(zip(*elist)))
            blist = [os.path.join(d, self.box_type, f) for f in
                     os.listdir(os.path.join(self.ds_path, d, self.box_type))]
            self.box_data.append(tuple(blist))

    @lru_cache()
    def get_example(self, i):
        """
            Get the i-th example
               Args:
                i (int): The index of the example
               Returns:
                a list of a tuple of feature array and box list
        """
        sample = []

        for j in range(len(self.feature_data[i])):
            f_paths = list(self.feature_data[i][j])
            for p, paths in enumerate(f_paths):
                f_path = os.path.join(self.ds_path, paths)
                b_path = os.path.join(self.ds_path, self.box_data[i][j])
                f_array = np.load(f_path)['arr_0']
                # resize the width and height according to layer - 20190209
                # downsampling of features (h,w == 13 remain, h,w == 26 -> halve to 13, h/w == 52 -> divide by 4) - 20190301
                # revert downsampling mode, choose only 1 layer to extract feature (Priority: conv33 -> conv39 -> conv45) - 20190408
                shape = list(f_array.shape)

                f_array.shape = tuple(shape)
                if p == 0:
                    f_arrays = f_array
                else:
                    f_arrays = np.concatenate([f_arrays, f_array], axis=0)

            with open(b_path, 'r', encoding='útf-8') as f:
                lines = f.readlines()
            b_list = []
            for line in lines:
                elements = line.strip().split()
                b_list.append([elements[0], float(elements[1]), float(elements[2]), float(elements[3]),
                               float(elements[4])])  # [label, center-x, center-y, width, height]
            sample.append((tuple(f_arrays), b_list))

        return self.dirs[i], sample


def __len__(self):
    """Get the length of a dataset
       Returns:
        len (int): length of the dataset (that is, the number of video clips)
    """
    return len(self.feature_data)


def get_layer_info(self):
    """Get layer information
       Returns:
        layer_info (tuple): a tuple of layer_name, height, width, channels
    """
    return self.layer_info


def get_feature_type(self):
    """Get feature type
       Returns:
        feature_type (str): feature type 'raw', 'tbox_processed' or 'ebox_processed'
    """
    return self.feature_type


def get_box_type(self):
    """Get box type
       Returns:
        box_type (str): box type 'tbox' or 'ebox'
    """
    return self.box_type


def get_length(self):
    """Get the length of a dataset
       Returns:
        len (int): length of the dataset (that is, the number of video clips)
    """
    return self.__len__()


def prepare_input_sequence(self, batch, roi_bg=('BG_ZERO')):
    """ Prepare input sequence
                Args:
                 batch (list of dataset samples): a list of samples of dataset
                Returns:
                 feature batch (list of arrays): a list of feature arrays
    """
    ffs_batch = []
    if self.feature_type == 'raw':
        for one_batch in batch:
            ffs = [element[0] for element in
                   one_batch[1]]  # a list of array[1 x channel x h x w], size=the length of sequence
            rbs = [element[1] for element in one_batch[1]]
            roi_ffs = []
            for i in range(len(ffs)):
                roi_ffs.append(self.extract_roi_feature(ffs[i], rbs[i], roi_bg))
            ffs_batch.append(roi_ffs)
    else:
        for one_batch in batch:
            ffs = [element[0] for element in one_batch]
            ffs_batch.append(ffs)
    # frame (ROI) feature batch sequence
    ffb_seq = []  # a sequence length list of frame feature

    for i in range(len(ffs_batch[0])):
        ffb_seq.append(np.concatenate([alist[i] for alist in ffs_batch]))

    return ffb_seq


def extract_roi_feature(self, feature, box, roi_bg):  # <ADD roi_bg/>
    """ Extract ROI feature
                Args:
                 feature (numpy array): feature array  (1, channel, height, width)
                 box (list): box information
                Returns:
                 extracted feature (numpy array): extracted feature array
    """
    # <MOD>
    bg = roi_bg[0]  # bg ::= BG_ZERO|BG_GN(Gaussian Noise)|BG_DP(Depression)
    if bg == 'BG_ZERO':
        # zero array with the same shape
        extracted_feature = np.zeros_like(feature)
    elif bg == 'BG_GN':
        gn_mean = 0.0
        gn_std = roi_bg[1]
        # extracted_feature = np.random.normal(gn_mean, gn_std, feature.shape)
        # convert feature into np array to use shape attr - 20190226
        np_feature = np.asarray(feature)
        extracted_feature = np.random.normal(gn_mean, gn_std, np_feature.shape)
    elif bg == 'BG_DP':
        depression = roi_bg[1]
        extracted_feature = feature * depression
    # </MOD>
    # partial substitution
    # l_name, l_height, l_width, l_channels = self.layer_info
    l_height, l_width = extracted_feature.shape[2:]
    feat_np = np.asarray(feature)
    for b in box:  # [label, center-x, center-y, width, height]
        x0 = b[1] - b[3] / 2
        y0 = b[2] - b[4] / 2
        x1 = b[1] + b[3] / 2
        y1 = b[2] + b[4] / 2
        l_x0 = math.floor(l_width * x0)
        l_y0 = math.floor(l_height * y0)
        l_x1 = math.ceil(l_width * x1)
        l_y1 = math.ceil(l_height * y1)
        # extracted_feature[:,:,l_y0:l_y1,l_x0:l_x1] = feature[:,:,l_y0:l_y1,l_x0:l_x1]
        extracted_feature[:, :, l_y0:l_y1, l_x0:l_x1] = feat_np[:, :, l_y0:l_y1, l_x0:l_x1]
    return extracted_feature

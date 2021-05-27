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
import albumentations as A
import torchvision.datasets as datasets

from albumentations.pytorch import ToTensorV2
from functools import lru_cache


class TripDataset():
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

    #     Continue here (5/27/2021)

    @lru_cache()
    def get_example(self, i):
        pass

    def __len__(self):
        """Get the length of a dataset
           Returns:
            len (int): length of the dataset (that is, the number of video clips)
        """
        pass

    def get_layer_info(self):
        """Get layer information
           Returns:
            layer_info (tuple): a tuple of layer_name, height, width, channels
        """
        pass

    def get_feature_type(self):
        """Get feature type
           Returns:
            feature_type (str): feature type 'raw', 'tbox_processed' or 'ebox_processed'
        """
        pass

    def get_box_type(self):
        """Get box type
           Returns:
            box_type (str): box type 'tbox' or 'ebox'
        """
        pass

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
        pass

    def extract_roi_feature(self, feature, box, roi_bg):  # <ADD roi_bg/>
        """ Extract ROI feature
                    Args:
                     feature (numpy array): feature array  (1, channel, height, width)
                     box (list): box information
                    Returns:
                     extracted feature (numpy array): extracted feature array
        """
        # <MOD>
        pass

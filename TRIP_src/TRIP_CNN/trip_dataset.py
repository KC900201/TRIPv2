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
    """
    @lru_cache()
    def __init__(self):
        pass

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

"""
@author: kcng
@filename: trip_c_lstm.py
@coding: utf-8
========================
Date          Comment
========================
05152021      First revision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from trip_lstm import TripLSTM

import TRIP_src.config

class TripCLSTM(TripLSTM):
    """
        A class of TRIP(Traffic Risk Prediction) model,
        which has a pooling layer on the input side
    """
    def __init__(self, input_size, hidden_size, model_arch):
        """
            Constructor
            Args:
                input_size (int):
                hidden_size (int):
                model_arch (str):
        """
        super(TripCLSTM, self).__init__(input_size, hidden_size)
        with self.init_scope():
            pass

    def __call__(self, x):
        """ Forward propagation
            Args:
             x (a Variable of feature array): a feature array
            Returns:
             h (a Variable of hidden state array): a hidden state array
        """
        dropout_ratio = 0.2
        pass


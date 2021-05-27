"""
@author: kcng
@filename: trip_c_lstm.py
@coding: utf-8
========================
Date          Comment
========================
05152021      First revision
05272021      Completed pytorch source code conversion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from trip_lstm import TripLSTM

import TRIP_src.config


def spatial_pyramid_pooling_2d(input, output_size):
    assert input.dim() == 4 and input.size(2) == input.size(3)
    return F.max_pool2d(input, kernel_size=input.size(2))  # output size


class TripCLSTM(TripLSTM):
    """
        A class of TRIP(Traffic Risk Prediction) model,
        which has a pooling layer on the input side
    """

    def __init__(self, input_size, hidden_size, model_arch='MP-C-RL-SPP4-LSTM'):
        """
            Constructor
            Args:
                input_size (int): an input size of LSTM
                hidden_size (int): a hidden size of LSTM
                model_arch (str): a model architecture
        """
        super(TripCLSTM, self).__init__(input_size, hidden_size)
        with self.init_scope():
            self.input_conv = nn.Conv2d(None, 512, kernel_size=3, stride=1, padding=1)
            self.input_middle_conv = nn.Conv2d(None, 512, kernel_size=3, stride=1, padding=1)
            self.input_sec_middle_conv = nn.Conv2d(None, 512, kernel_size=3, stride=1, padding=1)
            self.input_third_middle_conv = nn.Conv2d(None, 512, kernel_size=3, stride=1, padding=1)
        self.model_arch = model_arch

    # Self defined spatial pyramid pooling in Pytorch
    # Reference URL -> https://discuss.pytorch.org/t/elegant-implementation-of-spatial-pyramid-pooling-layer/831

    def __call__(self, x):
        """ Forward propagation
            Args:
             x (a Variable of feature array): a feature array
            Returns:
             h (a Variable of hidden state array): a hidden state array
        """
        dropout_ratio = 0.2
        if self.model_arch == 'MP-C-RL-SPP4-LSTM':
            z = F.max_pool2d(x, 2)
            z = F.relu(self.input_conv(z))
            z = spatial_pyramid_pooling_2d(z, 4)
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-RL-SPP4-DO-LSTM':
            z = F.max_pool2d(x, 2)
            z = F.relu(self.input_conv(z))
            z = spatial_pyramid_pooling_2d(z, 4)
            z = F.tanh(self.input(z))
            z = F.dropout(z, dropout_ratio=dropout_ratio)

        return self.lstm(z)

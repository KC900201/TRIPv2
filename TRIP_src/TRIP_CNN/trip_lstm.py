"""
@author: kcng
@filename: trip_lstm.py
@coding: utf-8
========================
Date          Comment
========================
05152021      First revision
05192021      Recoding using Pytorch library
"""

import torch
import cupy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TripLSTM(nn.Module):
    """
        A class of TRIP(Traffic Risk Prediction) model
    """

    def __init__(self, input_size, hidden_size):
        super(TripLSTM, self).__init__()
        with self.init_scope():
            self.input = nn.Linear(in_features=None, out_features=input_size)
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
            self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
            self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
            self.ho = nn.Linear(in_features=hidden_size, out_features=1)

    def __call__(self, x):
        """ Forward propagation
            Args:
             x (a Variable of feature array): a feature array
            Returns:
             h (a Variable of hidden state array): a hidden state array
        """
        z = F.tanh(self.input(x))
        return self.lstm(z)

    def predict_risk(self, x):
        """ Risk prediction
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variable of float): a risk value
        """
        # reset lstm state
        self.lstm.reset_parameters()
        # recurrent reasoning and risk prediction
        for t in range(len(x)):
            v = x # Need to modify this, chainer => Variable, how does pytorch replace this?
            h = self(v)
        return F.sigmoid(self.ho(h))

    def predict_mean_risk(self, x):
        """ Risk prediction (mean)
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variavle of float): a risk value
        """
        # reset lstm state
        pass
        # recurrent seasoning and risk prediction

    def predict_max_risk(self, x):
        """ Risk prediction (max)
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variable of float): a risk value
        """
        # reset lstm state
        pass
        # recurrent reasoning and risk prediction

    def predict_max_risk_2(self, x):
        """ Risk prediction (max)
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variable of float): a risk value
        """
        # reset lstm state
        pass

    def comparative_loss(self):
        """ Comparative loss function
            Args:
             ra (a Variable of float array): anchor risk (minibatch size)
             rc (a Variable of float array): comparison risk (minibatch size)
             rel (a numpy array of int {[1], [-1], [0]} array: if 'ra' element must be greater than 'rc' element, 'rel' element is [1].
                                                               if 'ra' element must be less than 'rc' element, 'rel' element is [-1].
                                                               if 'ra' element and 'rc' element must be equal, 'rel' element is [0].
             margin (float): margin of risk (a small value(<1.0))
            Returns:
             loss (Variable of float array): comparative loss
        """

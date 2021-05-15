"""
@author: kcng
@filename: trip_lstm.py
@coding: utf-8
========================
Date          Comment
========================
05152021      First revision
"""

import torch
import cupy
import torch.nn as nn
import numpy as np


class TripLSTM():
    """A class of TRIP(Traffic Risk Prediction) model
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """ Forward propagation
            Args:
             x (a Variable of feature array): a feature array
            Returns:
             h (a Variable of hidden state array): a hidden state array
        """
        pass

    def predict_risk(self, x):
        """ Risk prediction
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variable of float): a risk value
        """
        pass

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

"""
@author: kcng
@filename: trip_trainer.py
@coding: utf-8
========================
Date          Comment
========================
05152021      First revision
"""
import torch
import torch.cuda
import torchvision.datasets as datasets

import numpy as np
import time
import os

from trip_dataset import TripDataset
from trip_lstm import TripLSTM
from trip_c_lstm import TripCLSTM


class TripTrainer(object):
    """A class of TRIP(Traffic Risk Prediction) trainer
    """

    def __init__(self):
        pass

    def set_model(self):
        """ Set a model and its parameters
            Args:
             execution_mode (str): execution mode (train | retrain | test)
             model_path (str): a model parameter file path
        """
        pass

    def open_log_file(self):
        """
        Open a log file
        :return:
        """
        pass

    def close_log_file(self):
        """
        Close a log file
        :return:
        """
        pass

    def write_log_header(self):
        pass

    def learn_model(self):
        """
        Learning without trainer
        :return:
        """
        pass

    def learn_model_mix(self):
        """
        Learning without trainer (real + virtual dataset)
        :return:
        """
        pass

    def learn_model_virtual(self):
        """
        Learning without trainer (virtual dataset)
        :return:
        """
        pass

    def compare_risk_level(self, batch1, batch2, risk1, risk2):
        """
        Compare risk level
        :return:
            comparison result (numpy array): each element is one of {[1], [-1], [0]}
        """
        pass

    def test_model(self):
        """
        Test
        :return: none
        """
        self.evaluate('test')

    def evaluate(self, stage):
        """
        Evaluation
        :param stage: 'train' or 'test'
        :return: accuracy (%)
        """
        pass

    def evaluate_mix(self, stage):
        """
        Evaluation (real data + virtual data)
        @args: stage (str)
        :param stage: 'train' or 'test'
        :return: accuracy
        """
        pass

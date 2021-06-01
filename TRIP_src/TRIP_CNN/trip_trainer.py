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

    def __init__(self, train_ds_path1, train_spec_file_name1, train_risk1,
                 train_ds_path2, train_spec_file_name2, train_risk2,
                 test_ds_path1, test_spec_file_name1, test_risk1,
                 test_ds_path2, test_spec_file_name2, test_risk2,
                 # Hide virtual and mix training path for now
                 # vtrain_ds_path1, vtrain_spec_file_name1, vtrain_risk1,
                 # vtrain_ds_path2, vtrain_spec_file_name2, vtrain_risk2,
                 # mtrain_ds_path1, mtrain_spec_file_name1, mtrain_risk1,
                 # mtrain_ds_path2, mtrain_spec_file_name2, mtrain_risk2,
                 layer_name, box_type, execution_mode, num_of_epoch, minibatch_size,
                 eval_interval, save_interval, model_param_file_path, tlog_path, gpu_id,
                 skip_interval=0):
        """
            Constructor
            Args:
             train_ds_path1 (str): a train dataset path 1
             train_spec_file_name1 (str): a train dataset spec file name 1
             train_risk1 (int): risk level of train dataset 1 which is 0 for no-accident and 1 for accident
             train_ds_path2 (str): a train dataset path 2
             train_spec_file_name2 (str): a train dataset spec file name 2
             train_risk2 (int): risk level of train dataset 2 which is 0 for no-accident and 1 for accident
             test_ds_path1 (str): a test dataset path 1
             test_spec_file_name1 (str): a test dataset spec file name
             test_risk1 (int): risk level of test dataset 1 which is 0 for no-accident and 1 for accident
             test_ds_path1 (str): a test dataset path 2
             test_spec_file_name1 (str): a test dataset spec file name 2
             test_risk1 (int): risk level of test dataset 2 which is 0 for no-accident and 1 for accident
             vtrain_ds_path1 (str): a train dataset path 1 for virtual data
             vtrain_spec_file_name1 (str): a train dataset spec file name 1 for virtual data
             vtrain_risk1 (int): risk level of train dataset 1 which is 0 for no-accident and 1 for accident for virtual data
             vtrain_ds_path2 (str): a train dataset path 2 for virtual data
             vtrain_spec_file_name2 (str): a train dataset spec file name 2 for virtual data
             vtrain_risk2 (int): risk level of train dataset 2 which is 0 for no-accident and 1 for accident for virtual data
             vtest_ds_path1 (str): a test dataset path 1 for virtual data
             vtest_spec_file_name1 (str): a test dataset spec file name for virtual data
             vtest_risk1 (int): risk level of test dataset 1 which is 0 for no-accident and 1 for accident for virtual data
             vtest_ds_path1 (str): a test dataset path 2 for virtual data
             vtest_spec_file_name1 (str): a test dataset spec file name 2 for virtual data
             vtest_risk1 (int): risk level of test dataset 2 which is 0 for no-accident and 1 for accident for virtual data
             layer_name (str): a layer name
             execution_mode (str): execution mode (train | retrain | test)
             num_of_epoch (int): the number of epochs
             minibatch_size (int): the size of minibatch
             eval_interval (int): evaluation interval
             save_interval (int): save interval
             box_type (str): a type of boxes - 'tbox' or 'ebox'
             model_param_file_path (str): a model parameter file path
             tlog_path (str): a training log file path
             gpu_id (int): GPU ID (-1 for CPU)
        """

        # set dataset
        self.train_ds1 = TripDataset(train_ds_path1, train_spec_file_name1, layer_name, box_type, skip_interval)
        self.train_ds2 = TripDataset(train_ds_path2, train_spec_file_name2, layer_name, box_type, skip_interval)
        self.test_ds1 = TripDataset(test_ds_path1, test_spec_file_name1, layer_name, box_type, skip_interval)
        self.test_ds2 = TripDataset(test_ds_path2, test_spec_file_name2, layer_name, box_type, skip_interval)
        # set risk value
        self.train_risk1 = train_risk1
        self.train_risk2 = train_risk2
        self.test_risk1 = test_risk1
        self.test_risk2 = test_risk2

        #     check dataset
        if self.train_ds1.get_length() == self.train_ds2.get_length():
            self.train_ds_length = self.train_ds1.get_length()
        else:
            raise ValueError('Mismatch of training dataset length')

        if self.test_ds1.get_length() == self.test_ds2.get_length():
            self.test_ds_length = self.test_ds1.get_length()
        else:
            raise ValueError('Mismatch of testing dataset length')

        if (self.train_ds1.get_layer_info() != self.train_ds2.get_layer_info()) or (
                self.test_ds1.get_layer_info() != self.test_ds2.get_layer_info()):
            raise ValueError('Mismatch of layer info')

        if self.train_ds1.get_feature_type() != self.train_ds2.get_feature_type():
            raise ValueError('Mismatch of training feature types')

        if self.test_ds1.get_feature_type() != self.test_ds2.get_feature_type():
            raise ValueError('Mismatch of test feature types')

        if self.train_ds1.get_box_type() != self.train_ds2.get_box_type():
            raise ValueError('Mismatch of training box types')

        if self.test_ds1.get_box_type() != self.test_ds2.get_box_type():
            raise ValueError('Mismatch of test box types')

        # set gpu
        self.gpu_id = gpu_id

        if self.gpu_id >= 0:
            pass  # set up later (refer to trip_predictor.py)

        # set model
        self.set_model(execution_mode, model_param_file_path)

        # set epoch, minibatch, eval interval, save interval, training log file path <--- can set tensorboard
        self.num_of_epoch = num_of_epoch
        self.minibatch_size = minibatch_size
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.tlog_path = tlog_path
        self.tlogf = None
        self.previous_acc = 0
        self.previous_train_acc = 0

    def set_model(self, execution_mode, model_param_file_path):
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

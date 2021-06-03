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
import torch.cuda as cuda
import torch.optim as optim
import torchvision.datasets as datasets

import numpy as np
import time
import os
import TRIP_src.config as config

from torch import serialization, cuda
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
            self.device = cuda.device(self.gpu_id)  # set up later (refer to trip_predictor.py)
        else:
            self.device = torch.device(config.DEVICE)

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
        self.learning_rate = config.LEARNING_RATE
        self.weight_decay = config.WEIGHT_DECAY  # wait for review

    def set_model(self, execution_mode, model_param_file_path):
        """ Set a model and its parameters
            Args:
             execution_mode (str): execution mode (train | retrain | test)
             model_path (str): a model parameter file path
        """
        self.execution_mode = config.EXECUTION_MODE
        # default parameters
        self.model_path = './model/trip_model.npz'
        self.model_arch = 'MP-C-RL-SPP4-LSTM'  # default model arch = highest accuracy from TRIPv1
        self.input_size = config.INPUT_SIZE
        self.hidden_size = config.HIDDEN_SIZE
        self.roi_bg = config.ROI_BG
        self.comparative_loss_margin = config.COMPARISON_LOSS_MARGIN
        self.risk_type = config.RISK_TYPE
        self.threshold_of_similar_risk = config.THRESHOLD_SIMILAR_RISK
        self.optimizer_info = config.OPTIMIZER
        self.momentum = config.MOMENTUM

        # read parameters
        # bypass first, may not need param files

        # set model
        if self.model_arch == 'FC-LSTM':
            self.model = TripLSTM(self.input_size, self.hidden_size)
        else:
            self.model = TripCLSTM(self.input_size, self.hidden_size, self.model_arch)

        # set optimizer
        optimizer_info = self.optimizer_info.lower()
        if self.execution_mode == 'train' or self.execution_mode == 'retrain':
            if optimizer_info == 'train' or self.execution_mode.lower() == 'retrain':
                if optimizer_info == 'adam':
                    self.optimizer = optim.Adam(lr=self.learning_rate, weight_decay=self.weight_decay)
                elif optimizer_info == 'adadelta':
                    self.optimizer = optim.Adadelta(lr=self.learning_rate, weight_decay=self.weight_decay)
                elif optimizer_info == 'adagrad':
                    self.optimizer = optim.Adagrad(lr=self.learning_rate)
                elif optimizer_info in ('momentum_sgd', 'sgd'):
                    self.optimizer = optim.SGD(lr=self.learning_rate, momentum=self.momentum,
                                               weight_decay=self.weight_decay)
                elif optimizer_info == 'rmsprop':
                    self.optimizer = optim.RMSprop(lr=self.learning_rate)
                else:
                    raise ValueError('Illegal optimizer info')  # remove nesterovag, smorms3, rmspropgraves

            # set model to optimizer
            if self.execution_mode == 'train':  # Double check
                self.optimizer = self.optimizer.add_param_group(self.model.parameters())
            else:
                print('Loading a model: {}'.format(self.model_path))
                np.load(self.model_path, self.model)  # load a model
                self.optimizer = self.optimizer.add_param_group(self.model.parameters())  # set model to an optimizer
                np.load(self.model_path.replace('model.npz', 'optimizer.npz'), self.optimizer)  # load an optimizer
                print('done')
        else:  # test model
            print('Loading a model: {}'.format(self.model_path))  # Double check
            np.load(self.model_path, self.model)
            print(' done')

        if self.gpu_id >= 0:
            self.device = cuda.device(self.gpu_id)  # set up later (refer to trip_predictor.py)
        else:
            self.device = torch.device(config.DEVICE)

    def open_log_file(self):
        """
        Open a log file
        :return:
        """
        if self.execution_mode == 'train':
            self.tlogf = open(self.tlog_path, 'x', encoding='utf-8')  # 20190510
        elif self.execution_mode == 'retrain':
            self.tlogf = open(self.tlog_path, 'a', encoding='utf-8')

    def close_log_file(self):
        """
        Close a log file
        :return:
        """
        if self.tlogf is not None:
            self.tlogf.close()

    def write_log_header(self):  # Do we still need it if we have TensorBoard?
        self.tlogf.write('[Head]\n')
        self.tlogf.write('Train DS 1: {0}, {1}\n'.format(os.path.basename(self.train_ds1.ds_path), self.train_risk1))
        self.tlogf.write('Train DS 2: {0}, {1}\n'.format(os.path.basename(self.train_ds2.ds_path), self.train_risk2))
        self.tlogf.write('Test DS 1: {0}, {1}\n'.format(os.path.basename(self.test_ds1.ds_path), self.test_risk1))
        self.tlogf.write('Test DS 2: {0}, {1}\n'.format(os.path.basename(self.test_ds2.ds_path), self.test_risk2))
        self.tlogf.write('Train DS length: {}\n'.format(self.train_ds_length))
        self.tlogf.write('Test DS length: {}\n'.format(self.test_ds_length))
        # Layer info
        layer_info = self.train_ds1.get_layer_info()
        self.tlogf.write(
            'Layer: {0} ({1},{2},{3})\n'.format(layer_info[0], layer_info[1], layer_info[2], layer_info[3]))
        self.tlogf.write('Box type: {}\n'.format(self.train_ds1.get_box_type()))
        # Model architecture
        self.tlogf.write('Model arch: {}\n'.format(self.model_arch))
        # Input size and hidden size
        self.tlogf.write('Input size: {}\n'.format(self.input_size))
        self.tlogf.write('Hidden size: {}\n'.format(self.hidden_size))
        # ROI
        if len(self.roi_bg) == 2:
            bg = '(' + self.roi_bg[0] + ',' + str(self.roi_bg[1]) + ')'
        else:
            bg = '(' + self.roi_bg[0] + ')'
        self.tlogf.write('ROI BG: {}\n'.format(bg))
        # Hyperparameters
        self.tlogf.write('Risk type: {}\n'.format(self.risk_type))
        self.tlogf.write('Comparative loss margin: {}\n'.format(self.comparative_loss_margin))
        self.tlogf.write('Optimizer info: {}\n'.format(self.optimizer_info))
        self.tlogf.write('Minibatch size: {}\n'.format(self.minibatch_size))
        self.tlogf.write('Threshold of similar risk: {}\n'.format(self.threshold_of_similar_risk))
        self.tlogf.write('[Body]\n')
        self.tlogf.flush()

    def learn_model(self):
        """
        Learning without trainer
        :return:
        """
        self.open_log_file()
        self.write_log_header()

        # redefine the number of epochs (for retrain)
        start_epoch = self.optimizer
        num_of_epoch = self.num_of_epoch - start_epoch
        # Rewrite training process to suit Pytorch CNN flow (refer to Pytorch basics)
        # Continue 6/3/2021

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

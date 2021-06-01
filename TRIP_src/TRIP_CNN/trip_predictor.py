"""
@author: kcng
@filename: trip_predictor.py
@coding: utf-8
========================
Date          Comment
========================
05152021      First revision
"""

import os
import TRIP_src.config as config
import torch
from torch import serialization, cuda
import numpy as np

from trip_lstm import TripLSTM
from trip_c_lstm import TripCLSTM
from trip_dataset import TripDataset


class TripPredictor(object):
    def __init__(self, ds_path, spec_file_name, layer_name, box_type, window_size, model_param_file_path, plog_path,
                 gpu_id):
        """ Constructor
            Args:
             ds_path (str): a dataset path
             spec_file_name (str): a dataset spec file name
             layer_name (str): a layer name
             box_type (str): a type of boxes - 'tbox' or 'ebox'
             model_param_file_path (str): a model parameter file path
             gpu_id (int): GPU ID (-1 for CPU)
        """
        # set dataset
        self.ds = TripDataset(ds_path, spec_file_name, layer_name, box_type)

        # check dataset
        self.ds_length = self.ds.get_length()

        # set gpu
        self.gpu_id = gpu_id
        if self.gpu_id >= 0:
            self.device = cuda.device(self.gpu_id)
        else:
            self.device = torch.device(config.DEVICE)

        # window size
        self.window_size = window_size

        # set model
        self.set_model(model_param_file_path)

        # prediction log file path
        self.plog_path = plog_path
        self.plogf = None

    def set_model(self, model_param_file_path):
        """ Set a model and its parameters
            Args:
             model_path (str): a model parameter file path
        """
        # default parameters
        self.model_path = './model/trip_model.npz'
        # <ADD>
        self.model_arch = 'FC-LSTM'
        # </ADD>
        self.input_size = 1024
        self.hidden_size = 50
        # <ADD>
        self.roi_bg = ('ROI_BG')
        # </ADD>
        self.comparative_loss_margin = 0.05
        self.risk_type = 'seq_risk'
        # read parameters (Need to modify to use only best model)
        with open(model_param_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            param = line.split(':')
            if param[0].strip() == 'model_path':
                #                self.model_path = os.path.join(os.path.dirname(model_param_file_path), param[1].strip())
                npz_path = line.split(':')[1].strip().split()  # 10232019
                self.model_path = os.path.normpath(''.join(npz_path))
            # <ADD>
            elif param[0].strip() == 'model_arch':
                self.model_arch = param[1].strip()
            # </ADD>
            elif param[0].strip() == 'input_size':
                self.input_size = int(param[1].strip())
            elif param[0].strip() == 'hidden_size':
                self.hidden_size = int(param[1].strip())
            # <ADD>
            elif param[0].strip() == 'roi_bg':
                bg_param = param[1].strip().split()
                if len(bg_param) == 2:
                    bg_param[1] = float(bg_param[1])
                self.roi_bg = tuple(bg_param)
            # </ADD>
            elif param[0].strip() == 'comparative_loss_margin':
                self.comparative_loss_margin = float(param[1].strip())
            elif param[0].strip() == 'risk_type':  # { 'seq_risk' | 'seq_mean_risk' }
                self.risk_type = param[1].strip()
            else:
                continue
        # set model
        # <MOD>
        if self.model_arch == 'FC-LSTM':
            self.model = TripLSTM(self.input_size, self.hidden_size)
        else:
            self.model = TripCLSTM(self.input_size, self.hidden_size, self.model_arch)
        # </MOD>
        print('Loading a model: {}'.format(self.model_path))

        np.load(self.model_path, self.model)

        print(' done')
        if self.gpu_id >= 0: ## take note
            self.model.to(self.device)



    def open_log_file(self):
        """
        Open a log file
        :return: file log path
        """
        self.plogf = open(self.plog_path, 'w', encoding='utf-80')

    def close_log_file(self):
        """
         Close a log file
         :return: none
        """
        if self.plogf is not None:
            self.plogf.close()

    def write_log_header(self):
        self.plogf.write('[Head]\n')
        self.plogf.write('DS name: {}\n'.format(os.path.basename(self.ds.ds_path)))
        self.plogf.write('DS length: {}\n'.format(self.ds_length))
        layer_info = self.ds.get_layer_info()
        self.plogf.write(
            'Layer: {0} ({1},{2},{3})\n'.format(layer_info[0], layer_info[1], layer_info[2], layer_info[3]))
        self.plogf.write('Box type: {}\n'.format(self.ds.get_box_type()))
        self.plogf.write('Model: {}\n'.format(os.path.basename(self.model_path)))

        self.plogf.write('Model arch: {}\n'.format(self.model_arch))

        # </ADD>
        self.plogf.write('Input size: {}\n'.format(self.input_size))
        self.plogf.write('Hidden size: {}\n'.format(self.hidden_size))
        # <ADD>
        if len(self.roi_bg) == 2:
            bg = '(' + self.roi_bg[0] + ',' + str(self.roi_bg[1]) + ')'
        else:
            bg = '(' + self.roi_bg[0] + ')'

    def predict(self):
        """
        Prediction main function
        :return:
        """
        # open log file
        self.open_log_file()
        self.write_log_header()
        # prediction function (Contine 5/29/2021)

        # close log file
        self.close_log_file()

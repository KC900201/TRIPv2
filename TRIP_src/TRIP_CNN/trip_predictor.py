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
import TRIP_src.config

from trip_lstm import TripLSTM
from trip_c_lstm import TripCLSTM
from trip_dataset import TripDataset


class TripPredictor(object):
    def __init__(self):
        pass

    def set_model(self):
        pass

    def open_log_file(self):
        """
        Open a log file
        :return: file log path
        """
        pass

    def close_log_file(self):
        """
         Close a log file
         :return: none
        """
        pass

    def write_log_header(self):
        pass

    def predict(self):
        """
        Prediction main function
        :return:
        """
        # open log file
        self.open_log_file()
        self.write_log_header()
        # prediction function

        # close log file
        self.close_log_file()

"""
@author: kcng
@filename: yolov3_feature_extraction_model.py
@coding: utf-8
========================
Date          Comment
========================
06102021      First revision
"""

from __future__ import division

import itertools
import numpy as np

from base_feature_extraction_model import YOLOBase


class ResidualBlock():

    def __init__(self, *links):
        super(ResidualBlock, self).__init__(*links)

    def __call__(self, x):
        h = x
        for link in self:
            h = link(h)
        h += x
        return h


class Darknet53Extractor():

    def __init__(self):
        pass

    def __call__(self, x):
        pass


class YOLOv3(YOLOBase):
    pass

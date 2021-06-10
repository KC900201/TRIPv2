"""
@author: kcng
@filename: base_feature_extraction_model.py
@coding: utf-8
========================
Date          Comment
========================
06102021      First revision
"""

class YOLOBase():

    @property
    def insize(self):
        pass

    def use_preset(self, preset):
        pass

    def predict(self, imgs):
        pass
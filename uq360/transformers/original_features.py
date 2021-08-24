# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#


import numpy as np
from .feature_transformer import FeatureTransformer

class OriginalFeaturesTransformer(FeatureTransformer):
    def __init__(self):
        super(OriginalFeaturesTransformer, self).__init__()
    
    @classmethod
    def name(cls):
        return ('original_features')

    def transform(self, x, predictions):
        return x


    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass


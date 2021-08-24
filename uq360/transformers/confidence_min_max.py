# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#


import numpy as np
from .feature_transformer import FeatureTransformer

class ConfidenceMinMaxTransformer(FeatureTransformer):
    def __init__(self):
        super(ConfidenceMinMaxTransformer, self).__init__()
    
    @classmethod
    def name(cls):
        return ('confidence_min_max')

    def transform(self, x, predictions):
        return np.min(predictions, axis=1) / np.max(predictions, axis=1)

    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass
    
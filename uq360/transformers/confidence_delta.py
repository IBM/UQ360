# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#

import numpy as np
from .feature_transformer import FeatureTransformer

class ConfidenceDeltaTransformer(FeatureTransformer):
    def __init__(self):
        super(ConfidenceDeltaTransformer, self).__init__()

    @classmethod
    def name(cls):
        return ('confidence_delta')

    def transform(self, x, predictions):
        confs_sorted = np.sort(predictions) 
        conf_delta = confs_sorted[:,-1] - confs_sorted[:,-2]
        return conf_delta

    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass

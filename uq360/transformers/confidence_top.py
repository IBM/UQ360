# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#

import numpy as np
from .feature_transformer import FeatureTransformer

class ConfidenceTopTransformer(FeatureTransformer):
    def __init__(self):
        super(ConfidenceTopTransformer, self).__init__()
    
    @classmethod
    def name(cls):
        return ('confidence_top')

    def transform(self, x, predictions):
        confs_sorted = np.sort(predictions) 
        top_confs = confs_sorted[:,-1]
        return top_confs

    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass

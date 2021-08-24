# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#

import numpy as np
from .feature_transformer import FeatureTransformer

class PredictedClassTransformer(FeatureTransformer):
    def __init__(self):
        super(PredictedClassTransformer, self).__init__()
    
    @classmethod
    def name(cls):
        return ('predicted_class')

    def transform(self, x, predictions):
        return np.argmax(predictions, axis=1)

    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass

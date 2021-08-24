# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#

import numpy as np

from uq360.base import Base


class BatchFeature(Base):
    def __init__(self):
        pass

    # Initial setup fo the transformer
    def set_transformer(self, name, transformer):
        self.pointwise_type = name
        self.pointwise_transformer = transformer


    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass


    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):
        pass


    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):
        pass


    @classmethod
    def instance(cls, subtype_name=None, **params):
        subtype_name = subtype_name

        return super(BatchFeature, cls).instance(subtype_name, **params)



    

        



import json
import os
import pickle
import shutil

import numpy as np

from uq360.base import Base
from uq360.utils.calibrators.calibrator import Calibrator

"""
Base class for performance predictors. 

Fit stage: 
    Performance predictors use the test set and labels of the base/input model, 
    plus a set of derived/transformed features (see feature transformers in UQ360.transformers) in the fit step. 
    
Predict stage: 
    Using the original and derived features from a pool of unlabeled production data, returns pointwise confidences 
    and uncertainties: 
    confidences: probability that the base/input model will correctly predict the class of that datapoint. 
    uncertainties: performance predictor's estimate of its own prediction uncertainty
"""
class PerfPredictor(Base):
    def __init__(self, calibrator):
        self.random_state = 42
        self._object_registry = {}
        self.fit_status = False

        # A dictionary to stash any whitebox features the prediction has that can be used in the uncertainty model
        self.whitebox_features = {}

        if calibrator is None:
            self.calibrator = None
        else:
            self.calibrator = Calibrator.instance(calibrator)

    def fit(self, x_test, x_test_features, y_test):
        raise NotImplementedError("fit method is not implemented for predictor {}".format(self.name()))
        
    def predict(self, x, x_features):
        raise NotImplementedError("required method predict is not implemented for predictor {}".format(self.name()))

    @classmethod
    def instance(cls, subtype_name=None, **params):
        subtype_name = subtype_name
        return super(PerfPredictor, cls).instance(subtype_name, **params)

    # When asked to 'init' a whitebox feature, set it to zero (for now?)
    def init_whitebox_feature(self, feature_name):
        self.add_whitebox_feature(feature_name, 0, warn_if_not_initialized=False)

    def add_whitebox_feature(self, feature_name, value, warn_if_not_initialized=True):
        logged_name = self.name() + '.' + feature_name
        if warn_if_not_initialized and (logged_name not in self.whitebox_features):
            print('ERROR, whitebox feature:', logged_name, 'was not initialized before use')
        self.whitebox_features[logged_name] = value

    @staticmethod
    def unison_shuffle(x, y):
        assert len(x) == len(y)
        perm = np.random.permutation(len(y))
        return x[perm], y[perm]

    def _balance_data(self, train_x, train_y):
        num_correct_predictions = len(train_y[train_y == 1])
        num_incorrect_predictions = len(train_y[train_y == 0])
        try:
            if num_correct_predictions > num_incorrect_predictions:
                false_indices = np.where(train_y==0)[0]
                supplemental_set_x = train_x[false_indices]
                supplemental_set_y = train_y[false_indices]
                repeat_num = int(num_correct_predictions /
                                num_incorrect_predictions)
                remaining_num = num_correct_predictions - \
                    num_incorrect_predictions * repeat_num
            else:
                true_indices = np.where(train_y==1)[0]
                supplemental_set_x = train_x[true_indices]
                supplemental_set_y = train_y[true_indices]
                repeat_num = int(num_incorrect_predictions /
                                num_correct_predictions)
                remaining_num = num_incorrect_predictions - num_correct_predictions * repeat_num

            train_addition_x = list(supplemental_set_x)*(repeat_num-1)
            train_addition_y = list(supplemental_set_y)*(repeat_num-1)
            residual_indices = np.random.permutation(len(supplemental_set_y))[:remaining_num]
            new_train_x = np.concatenate((train_x, train_addition_x, supplemental_set_x[residual_indices]), axis=0)
            new_train_y = np.concatenate((train_y, train_addition_y, supplemental_set_y[residual_indices]), axis=0)
            new_train_x, new_train_y = self.unison_shuffle(new_train_x, new_train_y)
        except:
            print("Balancing data encountered a problem. Using unbalanced data.")
            return train_x, train_y
        
        return new_train_x, new_train_y

    @property
    def pkl_registry(self):
        pkl_list = []
        pkl_names = []
        for obj in self._object_registry.keys():
            if self._object_registry[obj]['type'] == 'pkl':
                pkl_list.append(self._object_registry[obj]['object'])
                pkl_names.append(obj)
        return pkl_list, pkl_names

    def register_pkl_object(self, obj, name):
        self._object_registry[name] = {'object': obj, 'type': 'pkl'}

    @property
    def json_registry(self):
        json_list = []
        json_names = []
        for obj in self._object_registry.keys():
            if self._object_registry[obj]['type'] == 'json':
                json_list.append(self._object_registry[obj]['object'])
                json_names.append(obj)
        return json_list, json_names

    def register_json_object(self, obj, name):
        self._object_registry[name] = {'object': obj, 'type': 'json'}

    def save(self, output_location=None):
        raise NotImplementedError("save method should be implemented by the predictor")

    def _save(self, output_location):
        assert os.path.isdir(output_location)
        if not self.fit_status:
            raise Exception("CANNOT SAVE PREDICTOR BEFORE CALLING FIT. ")

        # Append predictor name onto output path
        output_location = os.path.join(output_location, self.name())
        if os.path.isdir(output_location):
            print("WARNING: {} ALREADY EXISTS. OVERWRITING THIS DIRECTORY".format(output_location))
            shutil.rmtree(output_location)
            os.mkdir(output_location)
        else:
            os.mkdir(output_location)

        # Objects in separate registries are saved with different methods
        registers = self.pkl_registry
        for obj, name in zip(registers[0], registers[1]):
            filename = self.name() + '-' + name + '.pkl'
            with open(os.path.join(output_location, filename), 'wb') as f:
                pickle.dump(obj, f)

        registers = self.json_registry
        for obj, name in zip(registers[0], registers[1]):
            filename = self.name() + '-' + name + '.json'
            with open(os.path.join(output_location, filename), 'w') as f: 
                json.dump(obj, f)

        # If calibrator is not None, save
        if self.calibrator is not None:
            c_dir = os.path.join(output_location, 'calibrator-' + self.calibrator.name())
            if not os.path.isdir(c_dir):
                os.mkdir(c_dir)
            self.calibrator.save(c_dir)

    def load(self, input_location=None):
        raise NotImplementedError("load method should be implemented by the predictor")

    def _load(self, input_location):
        input_location = os.path.join(input_location, self.name())
        assert os.path.isdir(input_location)
        
        archived_files = []
        for r, _, f in os.walk(input_location):
            for item in f:
                if item.startswith(self.name()):
                    archived_files.append(os.path.join(r, item))

        for obj_file in archived_files:
            if obj_file.endswith('.pkl'):
                with open(obj_file, 'rb') as f:
                    obj = pickle.load(f)
                tail = os.path.split(obj_file)[1]
                name = tail.replace('.pkl','').replace(self.name()+'-','')
                self.register_pkl_object(obj, name)
            elif obj_file.endswith('.json'):
                with open(obj_file, 'r') as f:
                    obj = json.load(f)
                tail = os.path.split(obj_file)[1]
                name = tail.replace('.json','').replace(self.name()+'-','')
                self.register_json_object(obj, name)

        if self.calibrator is not None:
            c_dir = os.path.join(input_location, 'calibrator-' + self.calibrator.name())
            assert os.path.isdir(c_dir)
            self.calibrator.load(c_dir)

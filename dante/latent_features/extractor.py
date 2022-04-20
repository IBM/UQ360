import logging
import os
from functools import reduce
from typing import Callable, Union, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .latent_features import LatentFeatures


class ModelFeatureExtractor():

    def __init__(
        self,
        model: Callable,
        dataloaders: Tuple[DataLoader, str],
        layers: Union[List[List[str]], List[Tuple[List[str], str]]],
        model_name: str,
        out_dir: str = './output',
        loggin_level: str = 'DEBUG', 
        disable_tqdm = False):
    
        # Set loggin level
        self.logger = logging.getLogger('feature_extract')
        self.logger.setLevel(loggin_level)

        self.model = model
        self.dataloaders = dataloaders
        self.layers = layers
        self.model_name = model_name
        self.out_dir = out_dir
        self.disable_tqdm = disable_tqdm

    def batch_to_model_input(self, batch):
        x, _ = batch

        return x.cuda()

    def run(self):

        # If model_name is given, create subfolder
        if self.model_name is not None:
            out_dir = os.path.join(self.out_dir, self.model_name)
            os.mkdir(out_dir)
            model_name = self.model_name + '_'
        else:
            model_name = ''

        if type(self.layers[0]) is list:
            layers = [
                self.get_submodule(submodules_list, get_name=True) for
                submodules_list in self.layers
            ]
        elif type(self.layers[0]) is tuple:
            layers = [
                (self.get_submodule(submodules_list, get_name=False), l_name)
                for submodules_list, l_name in self.layers
            ]
        else:
            raise ValueError('layer argument format not supported.')
            
        # Extract latent and confidence
        for dl, dl_name in self.dataloaders:

            for layer, layer_name in layers:

                latent_features = self._extract_latent_features(dl, layer)
                
                fname = f"{model_name}latent_{dl_name}_{layer_name}.pt"
                torch.save(
                    latent_features,
                    os.path.join(out_dir, fname))
                
                self.logger.info(f"DONE: latent {dl_name}, dense layer {layer_name}")


    def _extract_latent_features(self, data_loader, layer):
        """Extract latent features of given layer.
        """

        latent_extractor = LatentFeatures(
            model=self.model, 
            layer=layer)

        latent_features = []
        for batch in tqdm(data_loader, disable=self.disable_tqdm):
            
            batch_input = self.batch_to_input(batch)

            lf = latent_extractor.extract(batch_input)[0]

            latent_features.append(lf)
        
        return torch.cat(latent_features)

    def get_submodule(self, submodules_list, get_name=False):

        module = reduce(
            lambda module, submodule_name: module.get_submodule(submodule_name), 
            [self.model] + submodules_list
        )

        if get_name:
            return module, '_'.join(submodules_list)
        else: 
            return module

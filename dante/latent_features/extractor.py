import logging
import os
from functools import reduce
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .latent_features import LatentFeatures


class ModelFeatureExtractor:
    def __init__(
        self,
        model: Callable,
        dataloader: Tuple[DataLoader, str],
        submodules_list: List[str],
        layer_name: str = None,
        model_name: str = None,
        out_dir: str = "./output",
        loggin_level: str = "DEBUG",
        disable_tqdm=False,
        allow_overwrite: bool = False,
    ):
        """_summary_

        Args:
            model (Callable): Pytorch model
            dataloader (Tuple[DataLoader, str]): Tuple (dataloader, name)
            submodules_list (List[str]): Submodule list to reach the desired
                layer. Call model to get the needed submodules.
            layer_name (str, optional): Given layer name. Defaults to None.
            model_name (str, optional): Given module name. Defaults to None.
            out_dir (str, optional): Output path. Defaults to './output'.
            loggin_level (str, optional): Logger level. Defaults to 'DEBUG'.
            disable_tqdm (bool, optional): Disable tqdm. Defaults to False.
            allow_overwrite (bool, optional): Overwrite existing file. Defaults to False.
        """

        # Set loggin level
        self.logger = logging.getLogger("feature_extract")
        self.logger.setLevel(loggin_level)

        self.model = model
        self.dataloader = dataloader
        self.submodules_list = submodules_list
        self.model_name = model_name
        self.layer_name = layer_name
        self.out_dir = out_dir
        self.disable_tqdm = disable_tqdm
        self.allow_overwrite = allow_overwrite

    def batch_to_model_input(self, batch):
        x, _ = batch

        return x.cuda()

    def run(self) -> Tensor:

        # If model_name is given, create subfolder
        if self.model_name is not None:
            out_dir = os.path.join(self.out_dir, self.model_name)

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            model_name = self.model_name + "_"
        else:
            model_name = ""

        layer, layer_name = self.get_submodule(self.submodules_list, get_name=True)

        # Use given layer_name if available
        layer_name = self.layer_name if self.layer_name else layer_name

        # Extract latent and confidence
        dl, dl_name = self.dataloader

        fname = f"{model_name}latent_{dl_name}_{layer_name}.pt"
        fpath = os.path.join(out_dir, fname)

        # Load activations if available
        if os.path.isfile(fpath) and not self.allow_overwrite:
            return torch.load(fpath)

        latent_features = self._extract_latent_features(dl, layer)

        print(latent_features.shape)

        return

        torch.save(latent_features, fpath)

        return latent_features

    def _extract_latent_features(self, data_loader, layer):
        """Extract latent features of given layer."""

        latent_extractor = LatentFeatures(model=self.model, layer=layer)

        latent_features = []
        for batch in tqdm(data_loader, disable=self.disable_tqdm):

            batch_input = self.batch_to_input(batch)

            lf = latent_extractor.extract(batch_input)[0]

            latent_features.append(lf)

        return torch.cat(latent_features)

    def get_submodule(self, submodules_list, get_name=False):

        module = reduce(
            lambda module, submodule_name: module.get_submodule(submodule_name),
            [self.model] + submodules_list,
        )

        if get_name:
            return module, "-".join(submodules_list)
        else:
            return module

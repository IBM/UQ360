import argparse
import json
import logging
import os
import sys
import torch

import numpy as np
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import (
    DrugAffinityDataset, ProteinProteinInteractionDataset
)
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer

class PrepareTitan():

    def __init__(self, model_path, logging_level='ERROR'):

        self.model_path = model_path

        self._set_logger(logging_level)

        self._process_params()

        self.device = get_device()

        self._load_languages()

    def _set_logger(self, logging_level):
        logging.basicConfig(stream=sys.stdout)
        logger = logging.getLogger()
        logger.setLevel(logging_level)

        self.logger = logger

    def _process_params(self):
        params_filepath = os.path.join(self.model_path, 'model_params.json')
        params = {}
        with open(params_filepath) as fp:
            params.update(json.load(fp))
        
        self.params = params

    def _load_languages(self, test=False):

        smiles_language = SMILESTokenizer.from_pretrained(self.model_path)
        smiles_language.set_encoding_transforms(
            randomize=None,
            add_start_and_stop=self.params.get('ligand_start_stop_token', True),
            padding=self.params.get('ligand_padding', True),
            padding_length=self.params.get('ligand_padding_length', True),
            device=self.device,
        )
        smiles_language.set_smiles_transforms(
            augment=False if test else self.params.get('augment_smiles', False),
            canonical=self.params.get('smiles_canonical', False),
            kekulize=self.params.get('smiles_kekulize', False),
            all_bonds_explicit=self.params.get('smiles_bonds_explicit', False),
            all_hs_explicit=self.params.get('smiles_all_hs_explicit', False),
            remove_bonddir=self.params.get('smiles_remove_bonddir', False),
            remove_chirality=self.params.get('smiles_remove_chirality', False),
            selfies=self.params.get('selfies', False),
            sanitize=self.params.get('sanitize', False)
        )
        
        if test:
            if self.params.get('receptor_embedding', 'learned') == 'predefined':
                protein_language = ProteinFeatureLanguage.load(
                    os.path.join(self.model_path, 'protein_language.pkl')
                )
            else:
                protein_language = ProteinLanguage.load(
                    os.path.join(self.model_path, 'protein_language.pkl')
                )
        else:
            if self.params.get('receptor_embedding', 'learned') == 'predefined':
                protein_language = ProteinFeatureLanguage(
                    features=self.params.get('predefined_embedding', 'blosum')
                )
            else:
                protein_language = ProteinLanguage()

        self.smiles_language = smiles_language
        self.protein_language = protein_language

    def get_data(
        self,
        affinity_filepath,
        receptor_filepath,
        ligand_filepath,
        test=True):

        # Check if ligand as SMILES or as aa
        ligand_name, ligand_extension = os.path.splitext(ligand_filepath)
        if ligand_extension == '.csv':
            self.logger.info(
                'ligand file has extension .csv \n'
                'Please make sure ligand is provided as amino acid sequence.'
            )
            dataset = ProteinProteinInteractionDataset(
                sequence_filepaths=[[ligand_filepath], [receptor_filepath]],
                entity_names=['ligand_name', 'sequence_id'],
                labels_filepath=affinity_filepath,
                annotations_column_names=['label'],
                protein_language=self.protein_language,
                amino_acid_dict='iupac',
                padding_lengths=[
                    self.params.get('ligand_padding_length', None),
                    self.params.get('receptor_padding_length', None)
                ],
                paddings=self.params.get('ligand_padding', True),
                add_start_and_stops=self.params.get('add_start_stop_token', True),
                augment_by_reverts=
                    self.params.get('augment_test_data', False) if test
                    else self.params.get('augment_protein', False),
                randomizes=
                    False if test 
                    else self.params.get('randomize', False),
            )

        elif ligand_extension == '.smi':
            self.logger.info(
                'ligand file has extension .smi \n'
                'Please make sure ligand is provided as SMILES.'
            )

            dataset = DrugAffinityDataset(
                drug_affinity_filepath=affinity_filepath,
                smi_filepath=ligand_filepath,
                protein_filepath=receptor_filepath,
                smiles_language=self.smiles_language,
                protein_language=self.protein_language,
                smiles_padding=self.params.get('ligand_padding', True),
                smiles_padding_length=self.params.get('ligand_padding_length', None),
                smiles_add_start_and_stop=self.params.get(
                    'ligand_add_start_stop', True
                ),
                smiles_augment=False,
                smiles_canonical=self.params.get('test_smiles_canonical', False),
                smiles_kekulize=self.params.get('smiles_kekulize', False),
                smiles_all_bonds_explicit=self.params.get(
                    'smiles_bonds_explicit', False
                ),
                smiles_all_hs_explicit=self.params.get('smiles_all_hs_explicit', False),
                smiles_remove_bonddir=self.params.get('smiles_remove_bonddir', False),
                smiles_remove_chirality=self.params.get(
                    'smiles_remove_chirality', False
                ),
                smiles_selfies=self.params.get('selfies', False),
                protein_amino_acid_dict=self.params.get(
                    'protein_amino_acid_dict', 'iupac'
                ),
                protein_padding=self.params.get('receptor_padding', True),
                protein_padding_length=self.params.get('receptor_padding_length', None),
                protein_add_start_and_stop=self.params.get(
                    'receptor_add_start_stop', True
                ),
                protein_augment_by_revert=False,
                device=self.device,
                drug_affinity_dtype=torch.float,
                backend='eager',
                iterate_dataset=True
            )

            self.logger.info(
                f'ligand_vocabulary_size  {self.smiles_language.number_of_tokens} '
                f'receptor_vocabulary_size {self.protein_language.number_of_tokens}.'
            )

        else:
            raise ValueError(
                f"Choose ligand_filepath with extension .csv or .smi, \
            given was {ligand_extension}"
            )
        
        str_test = 'test' if test else 'train'
        self.logger.info(f'{str_test} Dataset has {len(dataset)} samples.')

        loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.params['batch_size'],
                shuffle=False,
                drop_last=True,
                num_workers=self.params.get('num_workers', 0)
            )
        
        self.dataset = dataset

        return loader
    
    def get_model(self, model_type='bimodal_mca'):

        model_fn = self.params.get('model_fn', model_type)
        model = MODEL_FACTORY[model_fn](self.params).to(self.device)
        model._associate_language(self.smiles_language)
        model._associate_language(self.protein_language)

        model_file = os.path.join(
            self.model_path, 'weights', 'done_training_bimodal_mca.pt'
        )

        self.logger.info(f'looking for model in {model_file}')

        if os.path.isfile(model_file):
            self.logger.info('Found existing model, restoring now...')
            model.load(model_file, map_location=self.device)

            self.logger.info(f'model loaded: {model_file}')

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f'Number of parameters: {num_params}')

        return model




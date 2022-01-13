"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Generalization Measures for Out of Distribution Generalization
"""
import torch

from typing import List

from domainbed_measures.utils import batchnorms2convs_

from domainbed_measures.utils import UnionDataLoader
from domainbed_measures.utils import DeterministicFastDataLoader

from domainbed.algorithms import Algorithm


class GenMeasure(object):
    def __init__(self,
                 algorithm: Algorithm,
                 train_loaders: List[DeterministicFastDataLoader],
                 held_out_loaders: List[DeterministicFastDataLoader],
                 num_classes: int,
                 train_epochs: int = 100,
                 device: str = "cuda",
                 needs_training: bool = False,
                 use_eval_data: bool = False,
                 convert_bn_to_conv: bool = False,
                 data_args: dict = None,
                 train_union_loader_type: str = 'longest_padded'):
        self._algorithm = algorithm
        self._needs_training = needs_training
        self._convert_bn_to_conv = convert_bn_to_conv
        self._num_classes = num_classes
        self._data_args = data_args
        self._use_eval_data = use_eval_data
        self._train_epochs = train_epochs
        self._device = device
        self._train_union_loader_type = train_union_loader_type

        self._algorithm.to(device)

        self._train_loaders = train_loaders
        if self._train_loaders is not None:
            self.reset_train_loader()

        if self._use_eval_data == False:
            self._union_held_out_loader = None
        else:
            self._held_out_loader_list = held_out_loaders
            self.reset_held_out_loader()

        if convert_bn_to_conv == True:
            batchnorms2convs_(self._algorithm.network)

        # Set the algorithm in evaluation mode
        if self._needs_training:
            self._algorithm.train()
        else:
            self._algorithm.eval()

    def reset_train_loader(self):
        if self._train_loaders is not None:
            self._train_loader = UnionDataLoader(
                self._train_loaders,
                self._train_union_loader_type)

    def reset_held_out_loader(self):
        if self._use_eval_data == False:
            raise ValueError("Cannot use eval data")
        self._union_held_out_loader = UnionDataLoader(
            self._held_out_loader_list)

    def _calculate_measure(self):
        raise NotImplementedError("Please implement in derived class")

    def compute(self, **kwargs):
        if self._needs_training == True:
            return self._calculate_measure(**kwargs)

        with torch.no_grad():
            return self._calculate_measure(**kwargs)
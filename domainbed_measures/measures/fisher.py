"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Generalization measures based on fisher information
"""
import numpy as np
import os
import pickle
import logging
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm

from domainbed_measures.measures.gen_measures import GenMeasure
from domainbed_measures.third_party.fisher_utils import StochasticFisher


class FisherEigValues(GenMeasure):
    _SEED = 42

    def __init__(self, *args, num_eig=100, max_num_examples=1000, **kwargs):
        super(FisherEigValues, self).__init__(*args, **kwargs)

        if num_eig >= max_num_examples or num_eig == -1:
            logging.warn("Capping num_eig to max number of examples")
            num_eig = max_num_examples

        self._stochastic_fisher = StochasticFisher(
            seed=self._SEED,
            max_num_examples=max_num_examples,
            num_eig=num_eig,
        )

    def _compute_train_test_fisher(self):
        logging.info("Computing training eigenvalues and vectors")
        train_eigval, train_eigvec = self._stochastic_fisher._compute(
            self._algorithm,
            self._train_loader,
        )

        logging.info("Computing heldout eigenvalues and vectors")
        heldout_eigval, heldout_eigvec = self._stochastic_fisher._compute(
            self._algorithm,
            self._union_held_out_loader,
        )
        return train_eigval, train_eigvec, heldout_eigval, heldout_eigvec

    def _calculate_measure(self):
        train_eigval, _, heldout_eigval, _ = self._compute_train_test_fisher()
        return heldout_eigval.sum() / train_eigval.sum(), {}


class FisherEigValuesSumDiff(FisherEigValues):
    _SEED = 42

    def _calculate_measure(self):
        train_eigval, _, heldout_eigval, _ = self._compute_train_test_fisher()
        return heldout_eigval.sum() - train_eigval.sum(), {}


class FisherEigVecAlign(FisherEigValues):
    _SEED = 42

    def _calculate_measure(self):
        _, train_eigvec, _, heldout_eigvec = self._compute_train_test_fisher()

        # L2 normalize train_eigvec
        train_eigvec = train_eigvec.T
        train_eigvec /= np.expand_dims(norm(train_eigvec, axis=-1), 1)
        heldout_eigvec = heldout_eigvec.T
        heldout_eigvec /= np.expand_dims(norm(heldout_eigvec, axis=-1), 1)

        similarity_matrix = heldout_eigvec.dot(train_eigvec.T)
        cost_matrix = -1 * similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return similarity_matrix[row_ind, col_ind].sum(), {}
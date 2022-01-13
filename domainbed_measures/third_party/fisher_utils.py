"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
 
Some utilities for computing fisher information

Author: Jonathan Frankle
"""
import numpy as np
from scipy.sparse.linalg import eigsh
import torch
from torch import autograd
import logging

from domainbed_measures.third_party import numpy_utils


def _compute_fisher_gradients(model,
                              data_loader,
                              max_num_examples=None,
                              num_classes=None,
                              masks=None):
    gradients = []
    model.train()

    params_dict = {k: v for k, v in model.named_parameters()}
    names, params = zip(*params_dict.items())

    examples_so_far = 0
    for idx, (x, _) in enumerate(data_loader):
        # Network inputs.
        x_var = torch.autograd.Variable(x).cuda()
        outputs = torch.nn.functional.softmax(model.predict(x_var), dim=1)
        max_num_examples_in_batch, num_outputs = list(outputs.shape)

        for i in range(max_num_examples_in_batch):
            # Early function exit condition.
            examples_so_far += 1
            if max_num_examples is not None and examples_so_far > max_num_examples:
                return gradients, examples_so_far

            # Potentially subsample the classes.
            num_classes = num_classes or num_outputs
            classes_to_use = np.concatenate(
                [np.ones(num_classes),
                 np.zeros(num_outputs - num_classes)])
            np.random.shuffle(classes_to_use)

            # Compute the per-class outputs.
            for use_class, output, sqrt_output in zip(classes_to_use,
                                                      outputs[i],
                                                      torch.sqrt(outputs[i])):
                if not use_class: continue

                grad_dict = dict(
                    zip(names, autograd.grad(output, params,
                                             create_graph=True)))
                grad_dict = {
                    k: v.clone().detach().cpu().numpy()
                    for k, v in grad_dict.items()
                }
                if masks is not None:
                    for k, v in masks.items():
                        grad_dict[k] = grad_dict[k][v == 1]
                gradients.append(
                    numpy_utils.vectorize(grad_dict) /
                    float(sqrt_output.item()))

            if (examples_so_far + 1) % 20 == 0:
                logging.info(
                    f"Finished {examples_so_far + 1}/{max_num_examples}")

    return gradients, examples_so_far


def _fisher_approximation_matrices(model,
                                   data_loader,
                                   max_num_examples=None,
                                   num_classes=None,
                                   masks=None):

    gradients, N = _compute_fisher_gradients(model, data_loader,
                                             max_num_examples, num_classes,
                                             masks)
    A = np.array(gradients)
    return A, A.dot(A.T) / N


class StochasticFisher(object):
    def __init__(
        self,
        seed: int,
        num_eig: int,
        max_num_examples: int = None,
        num_classes: int = None,
    ):
        self.seed = seed
        self.max_num_examples = max_num_examples
        self.num_eig = num_eig
        self.num_classes = num_classes

    def _compute(self, model, data_loader, masks=None):
        A, F_tilde = _fisher_approximation_matrices(
            model,
            data_loader,
            num_classes=self.num_classes,
            max_num_examples=self.max_num_examples,
            masks=masks)
        if not np.array_equal(F_tilde, F_tilde.T):
            raise ValueError('Not symmetric')

        logging.info("Computing the eigen values and eigen vectors")
        eigenvalues, eigenvectors = eigsh(F_tilde, self.num_eig)
        eigenvectors = A.T.dot(eigenvectors)

        return eigenvalues, eigenvectors
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Classical Generalization Measures for Supervised Learning
"""
import torch
import numpy as np
import copy
import math
import logging

from torch.nn.functional import softmax

from sklearn.metrics import accuracy_score

from skorch.dataset import unpack_data

from domainbed_measures.utils import clip_perturbated_param_
from domainbed_measures.utils import add_noise_to_param_
from domainbed_measures.utils import clone_trainer

from domainbed_measures.measures.gen_measures import GenMeasure

from domainbed_measures.utils import convert_algorithm_to_trainer
from domainbed_measures.utils import NegCrossEntropyLoss


class EntropyOutput(GenMeasure):
    def __init__(self, *args, compute_on="train", **kwargs):
        super(EntropyOutput, self).__init__(*args, **kwargs)
        if compute_on not in ['train', 'held_out']:
            raise ValueError
        self._compute_on = compute_on

    def entropy(self, dataloader):
        # Iterate for 1 epoch
        all_entropies = []

        for x, y in dataloader:
            x = x.to(self._device)
            y = y.to(self._device)
            algorithm_probs = softmax(self._algorithm.predict(x), dim=-1)
            entropy = -1 * torch.mean(
                torch.sum(algorithm_probs * torch.log(algorithm_probs),
                          dim=-1))

            all_entropies.append(float(entropy.cpu().float().numpy()))

        return np.mean(all_entropies)

    def _calculate_measure(self):
        if self._compute_on == 'train':
            return self.entropy(self._train_loader), {}
        return self.entropy(self._union_held_out_loader), {}


# The following code is heavily borowed from the paper
# Learning Optimal Representations with the Decodable Information Bottleneck
class PathNorm(GenMeasure):
    def _calculate_measure(self):
        """Compute the pathnorm as described in "FANTASTIC GENERALIZATION MEASURES AND WHERE TO FIND THEM".
        I.e. squares all parameters, foreward pass all ones and then take sqrt of output."""
        algorithm_copy = copy.deepcopy(self._algorithm)
        with torch.no_grad():
            for _, W in algorithm_copy.named_parameters():
                W.pow_(2)
        datum = self._train_loaders[0]._infinite_iterator._dataset[0][
            0].unsqueeze(0)
        all_ones = torch.ones_like(datum).to(self._device)
        logits = algorithm_copy.predict(all_ones)[0]
        sum_logits = logits.sum().cpu().item()

        return sum_logits**0.5, {}


class SharpnessMagnitude(GenMeasure):
    def __init__(self,
                 algorithm,
                 train_loaders,
                 held_out_loaders,
                 num_classes,
                 target_deviation=0.1,
                 n_restart_perturbate=3,
                 max_binary_search=50,
                 is_relative=True,
                 convert_bn_to_conv=True,
                 needs_training=True,
                 **kwargs):
        """
        Sharpness magnitude implemetnation mimicking:
            Jiang, Yiding, et al. "Fantastic Generalization Measures and Where to Find Them."
            arXiv print arXiv:1912.02178 (2019).

        Args:
            algorithm: DomainBed.algorithms.algorithm
            train_loaders: list of torch.utils.data.DataLoader
            held_out_loaders: list of torch.utils.data.DataLoader
            n_adv_perturbate: int, optional
                Number of steps to perform adversarial perturbation for.
            n_restart_perturbate: int, optional
                Number of times restarting the perturbation (different initialization for adv perturbate).
            target_deviation: float, optional
                Maximum difference of log likelihood allowed.
            max_binary_search: int, optional
                Maximum number of binary search tries
        """
        super(SharpnessMagnitude,
              self).__init__(algorithm,
                             train_loaders,
                             held_out_loaders,
                             num_classes,
                             convert_bn_to_conv=convert_bn_to_conv,
                             needs_training=needs_training,
                             **kwargs)

        self._target_deviation = target_deviation
        self._n_restart_perturbate = n_restart_perturbate
        self._max_binary_search = max_binary_search
        self._is_relative = is_relative
        self._measure_criterion = NegCrossEntropyLoss

        self._trainer_current = convert_algorithm_to_trainer(
            algorithm=self._algorithm,
            device=self._device,
        )

    def accuracy_trainer(self, trainer, dataloader):
        accuracy_all = []
        for x, y in dataloader:
            x = x.to(self._device)
            y = y.to(self._device)
            y_pred = trainer.predict_proba(x)
            accuracy_all.append(
                accuracy_score(y.cpu().numpy(), y_pred.argmax(-1)))

        self.reset_train_loader()
        return np.mean(accuracy_all)

    def get_sharp_mag_interval(
        self,
        unperturbed_trainer,
        unperturbed_acc,
        sigma_min,
        sigma_max,
    ):
        sigma_new = (sigma_min + sigma_max) / 2
        worst_acc = math.inf

        unperturbed_params = {
            name: param.detach()
            for name, param in unperturbed_trainer.module_.named_parameters()
        }

        for _ in range(self._n_restart_perturbate):
            trainer = clone_trainer(unperturbed_trainer,
                                    is_reinit_besides_param=True)

            # add half of the possible noise to give some space for gradient ascent
            add_noise_to_param_(trainer.module_,
                                sigma=sigma_new / 2,
                                is_relative=self._is_relative)

            self.reset_train_loader()
            for data in self._train_loader:
                Xi, yi = unpack_data(data)
                Xi = Xi.to(self._device)
                yi = yi.to(self._device)
                step = trainer.train_step(Xi, yi)

                # clipping perturbation value of added parameters to |w_i * sigma| or |sigma|
                clip_perturbated_param_(
                    trainer.module_,
                    unperturbed_params,
                    sigma_new,
                    is_relative=self._is_relative,
                )

                if not torch.isfinite(step["loss"]) or step["loss"].detach(
                ).abs() > (abs(unperturbed_acc) + 10 * self._target_deviation):
                    # if loss is very large for one batch then no need to finish this loop
                    return sigma_min, sigma_new
            self.reset_train_loader()

            curr_acc = self.accuracy_trainer(trainer, self._train_loader)
            worst_acc = min(worst_acc, curr_acc)

        deviation = abs(curr_acc - worst_acc)

        if math.isclose(unperturbed_acc, worst_acc, rel_tol=1e-2):
            # if not deviation is nearly zero can stop
            return sigma_new, sigma_new

        if deviation > self._target_deviation:
            sigma_max = sigma_new
        else:
            sigma_min = sigma_new

        return sigma_min, sigma_max

    def _calculate_measure(self, sigma_max=2.0, sigma_min=0.0):
        """
        Compute the sharpness magnitude 1/alpha'^2 described in [1].

        Notes
        -----
        - This is slightly different than [1] because the target deviation is
        on cross-entropy instead of accuracy

        Args:
            sigma_max: float, optional
            sigma_min: float, optional
                Minimum standard deviation of perturbation.
        """
        trainer = clone_trainer(self._trainer_current)
        trainer.criterion = self._measure_criterion
        trainer.initialize()

        acc = self.accuracy_trainer(trainer, self._train_loader)
        logging.info(f"Accuracy of original model: {acc}")

        for bin_search in range(self._max_binary_search):
            sigma_min, sigma_max = self.get_sharp_mag_interval(
                trainer,
                acc,
                sigma_min,
                sigma_max,
            )

            if sigma_min > sigma_max or math.isclose(
                    sigma_min, sigma_max, rel_tol=1e-2):
                # if interval for binary search is very small stop
                break

        if bin_search == self._max_binary_search - 1:
            logging.info(
                f"Stopped early beacuase reached max_binary_search={self._max_binary_search}.\
                [sigma_min,sigma_max]=[{sigma_min},{sigma_max}]")

        return 1 / (sigma_max**2), {}

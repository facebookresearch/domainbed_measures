"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import numpy as np
import copy
import math
import logging
import skorch
import os
import pickle

from collections import defaultdict
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch import nn

from skorch.callbacks import LRScheduler

from domainbed.lib.misc import accuracy as accuracy_fn

from domainbed_measures.measures.gen_measures import GenMeasure
from domainbed_measures.measures.losses import NegHDelHCriterion
from domainbed_measures.third_party.mmd_utils import mmd

from domainbed_measures.utils import ConcatModule

from domainbed_measures.utils import get_exponential_decay_gamma
from domainbed_measures.utils import permute_dataset
from domainbed_measures.utils import BaseRepIthDigits
from domainbed_measures.utils import TrainEvalNeuralNet
from domainbed_measures.utils import convert_algorithm_to_trainer
from domainbed_measures.utils import _freeze_params
from domainbed_measures.utils import _unfreeze_params
from domainbed_measures.utils import reset_weights_init


class FunctionVMinimality(GenMeasure):
    def __init__(self,
                 algorithm,
                 train_loaders,
                 held_out_loaders,
                 num_classes,
                 cond_min=False,
                 v_plus=False,
                 recompute_features_every_epoch=False,
                 **kwargs):
        super(FunctionVMinimality,
              self).__init__(algorithm, train_loaders, held_out_loaders,
                             num_classes, **kwargs)
        self._cond_min = cond_min
        self._v_plus = v_plus
        self._recompute_features_every_epoch = recompute_features_every_epoch

        self._trainer_current = convert_algorithm_to_trainer(
            algorithm=self._algorithm,
            device=self._device,
        )
        self._trainer_current.criterion = nn.CrossEntropyLoss
        self._trainer_current.initialize()

        self._base_rep_ith_digit = BaseRepIthDigits(self._num_classes)

    def get_num_heads(self, max_idx):
        return math.ceil(np.log(max_idx) / np.log(self._num_classes))

    def get_reinit_heads(self,
                         num_heads,
                         trained_classifier,
                         feat_dim,
                         batch_size=256,
                         lr=0.0025,
                         callbacks=None,
                         train_split=None):
        logging.info(f"Using batch size {batch_size}, lr {lr}")
        if callbacks == None:
            callbacks = []

        heads = []
        for hidx in range(num_heads):
            this_head = copy.deepcopy(trained_classifier)
            if self._v_plus == True:
                this_head = nn.Sequential(nn.Linear(feat_dim, feat_dim),
                                          nn.ReLU(), this_head)
            this_head.apply(reset_weights_init)

            this_head = TrainEvalNeuralNet(
                module=this_head,
                device=self._device,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                optimizer__lr=lr,
                batch_size=batch_size,
                max_epochs=self._train_epochs,
                train_split=
                train_split,  # None means no validation, only training
                iterator_train__shuffle=True,
                callbacks=callbacks)

            heads.append(this_head)

        return heads

    def compute_features(self, loader, featurizer, device):
        all_feats = []
        all_labels = []
        all_idx = []
        all_perclass_idx = []

        for x, y, idx in loader:
            feat_x = featurizer(x.to(device))
            all_feats.append(feat_x)
            all_labels.append(y.to(device))
            all_idx.append(idx.to(device))

        all_per_class_counts = defaultdict(int)
        all_labels = torch.cat(all_labels)
        for l in all_labels:
            l = int(l.cpu().data)
            all_perclass_idx.append(all_per_class_counts[l])
            all_per_class_counts[l] += 1

        self.reset_train_loader()

        return (torch.cat(all_feats), all_labels, torch.cat(all_idx),
                torch.Tensor(all_perclass_idx))

    def get_cond_v_entropy(self, heads, all_feats, all_idx, label_idx):
        # Do the optimization for each of the heads
        head_losses = []
        accumulated_head_epoch_losses = []
        for head_idx, head in enumerate(heads):
            # Recompute the features for each head, so that we are
            # closer to the stochastic setting for things like
            # dropout and so on.
            logging.info(f"Processing head {head_idx}/{len(heads)}")
            head.initialize()
            head.notify('on_train_begin')

            accumulated_loss = 0
            for ep in range(self._train_epochs):
                if self._recompute_features_every_epoch == True:
                    # Only way this can work is if shuffling is off in the loader
                    logging.info(f"Recomputing features..")
                    all_feats, all_idx = self.get_v_min_data(label_idx)

                all_this_head_labels = torch.Tensor(
                    self._base_rep_ith_digit(all_idx.cpu().numpy(),
                                             head_idx)).to(
                                                 self._device).long()

                if all_feats.shape[0] != all_this_head_labels.shape[0]:
                    raise ValueError("Shapes must match")

                dtrain, dval = head.get_split_datasets(all_feats,
                                                       all_this_head_labels)
                on_epoch_kwargs = {
                    "dataset_train": dtrain,
                    "dataset_valid": dval
                }

                head.notify("on_epoch_begin", **on_epoch_kwargs)
                head.run_single_epoch(dtrain,
                                      training=True,
                                      prefix="train",
                                      step_fn=head.train_step)
                head.notify("on_epoch_end", **on_epoch_kwargs)
                accumulated_loss += head.mean_train_loss(
                    all_feats, all_this_head_labels)

            # Run evaluation on the training set to find out the
            # loss on the training overall
            head_losses.append(
                head.mean_train_loss(all_feats, all_this_head_labels))
            accumulated_head_epoch_losses.append(accumulated_loss)

        return np.mean(head_losses), np.mean(accumulated_head_epoch_losses)

    def get_v_min_data(self, label_idx):
        featurizer = self._trainer_current.get_featurizer()
        _freeze_params(featurizer)
        logging.info("Precomputing features")
        all_feats, all_labels, all_idx, all_perclass_idx = self.compute_features(
            self._train_loader, featurizer, self._device)

        # Unconditional v-minimality
        if label_idx == -1:
            return all_feats, all_idx

        idx_with_label = [
            idx for idx, this_label in enumerate(all_labels)
            if this_label == label_idx
        ]
        feats_for_label = all_feats[idx_with_label, :]
        idx_for_label = all_perclass_idx[idx_with_label]

        _unfreeze_params(featurizer)
        return feats_for_label, idx_for_label

    def _calculate_measure(self,
                           num_head_batches=10,
                           max_lr=0.10,
                           lr_sweep_factor=0.5,
                           batch_size=256):
        """Calculate v-minimality.

        Args:
          num_head_batches: Number of batches of heads to optimize to improve the
            estimate of v-minimality
        """
        # Parameters related to learning
        #lr = self._algorithm.hparams['lr']
        #batch_size = self._algorithm.hparams['batch_size']
        callbacks = [
            LRScheduler(
                torch.optim.lr_scheduler.ExponentialLR,
                gamma=get_exponential_decay_gamma(100, self._train_epochs),
            )
        ]
        if self._cond_min == True:
            labels_to_process = range(self._num_classes)
        else:
            labels_to_process = [-1]

        v_entropy_x_z = []
        for label_idx in labels_to_process:
            # Prepare data per label
            task_feats, task_idx = self.get_v_min_data(label_idx)
            feat_dim = task_feats.shape[-1]

            # Get the heads for optimization
            num_heads = self.get_num_heads(int(torch.max(task_idx)))
            cond_v_entropy_across_batches = []
            accumulated_v_entropy_across_batches = []
            for batch_idx in range(num_head_batches):
                this_heads = (self.get_reinit_heads(
                    num_heads,
                    trained_classifier=(
                        self._trainer_current.get_classifier()),
                    feat_dim=feat_dim,
                    lr=max_lr * lr_sweep_factor**batch_idx,
                    batch_size=batch_size,
                    callbacks=callbacks))
                logging.info(
                    f"******** Batch of heads {batch_idx}/{num_head_batches}**********"
                )
                cond_v_entropy, accumulated_v_entropy = (
                    self.get_cond_v_entropy(this_heads, task_feats, task_idx,
                                            label_idx))
                cond_v_entropy_across_batches.append(cond_v_entropy)
                accumulated_v_entropy_across_batches.append(
                    accumulated_v_entropy)

            best_head_idx = np.argmin(accumulated_v_entropy_across_batches)
            v_entropy_x_z.append(cond_v_entropy_across_batches[best_head_idx])

        v_entropy_x_z = np.mean(v_entropy_x_z)
        return np.log(self._num_classes) - v_entropy_x_z


class ClassifierTwoSampleTest(FunctionVMinimality):
    """A classifier based two sample test, inspired by previous work:

    REVISITING CLASSIFIER TWO-SAMPLE TESTS
    David Lopez-Paz, Maxime Oquab
    ICLR 2017
    """
    def __init__(self, *args, per_env=False, train_or_test='test', **kwargs):
        super(ClassifierTwoSampleTest, self).__init__(*args, **kwargs)
        self._per_env = per_env
        self._train_or_test = train_or_test

    def get_heads(self,
                  num_batch_heads,
                  feat_dim,
                  num_labels,
                  criterion,
                  batch_size=256,
                  max_lr=0.0025,
                  lr_sweep_factor=0.5,
                  weight_decay_max=1e-1,
                  weight_decay_min=1e-4,
                  callbacks=None,
                  train_split=None):
        """Get heads for optimization.

        Set the maximum learning rate to start from for each head, and multiple
        heads then have a learning rate that is lr_sweep_factor^(i-1) * max_lr
        for the i'th head that we consider.
        """
        if callbacks == None:
            callbacks = []

        heads = []
        for hidx in range(num_batch_heads):
            lr = max_lr * lr_sweep_factor**(hidx)
            weight_decay = float(torch.multinomial(torch.logspace(1, -3, 5),
                                                   1))

            logging.info(
                f"Creating head {hidx} with lr {lr}, weight decay {weight_decay}"
            )
            if self._v_plus == True:
                this_head = nn.Sequential(
                    nn.Linear(feat_dim, int(feat_dim / 2)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 2), int(feat_dim / 4)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 4), int(feat_dim / 4)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 4), num_labels))
            else:
                this_head = nn.Linear(feat_dim, num_labels)

            this_head = TrainEvalNeuralNet(
                module=this_head,
                device=self._device,
                criterion=criterion,
                optimizer=torch.optim.SGD,
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=self._train_epochs,
                train_split=
                train_split,  # None means no validation, only training
                iterator_train__shuffle=True,
                callbacks=callbacks)

            heads.append(this_head)

        return heads

    @staticmethod
    def prepare_c2st_datasets(X_1, y_1, X_2, y_2):
        num_datapoints_per_label = min(X_1.shape[0], X_2.shape[0])

        X_1 = X_1[:num_datapoints_per_label, :]
        y_1 = y_1[:num_datapoints_per_label]
        X_2 = X_2[:num_datapoints_per_label, :]
        y_2 = y_2[:num_datapoints_per_label]

        X = torch.cat([X_1, X_2], dim=0)
        y = torch.cat([y_1, y_2], dim=0)
        X, y = permute_dataset(X, y)

        return X, y

    @staticmethod
    def accuracy_fn(net, X, y=None):
        dataloader = DataLoader(X, batch_size=256)
        correct = 0
        total = 0
        training = net.module_.training

        net.module_.eval()
        for this_X, this_y in dataloader:
            correct += net.module_.forward(this_X).argmax(1).eq(
                this_y).sum().item()
            total += this_X.shape[0]

        if training:
            net.module_.train()

        return float(correct) / total

    @staticmethod
    def convert_domain_classifier_accuracy_to_divergence(accuracy):
        error = 1 - accuracy
        return 2 * (1 - 2 * error)

    def _calculate_divergence_measure(self,
                                      all_train_feats,
                                      train_domain_labels,
                                      all_held_out_feats,
                                      heldout_domain_labels,
                                      lr_decay_gamma,
                                      num_head_batches,
                                      max_lr,
                                      lr_sweep_factor,
                                      train_env_to_use,
                                      train_val_split,
                                      trainval_test_split=0.8):

        all_feats, all_labels = self.prepare_c2st_datasets(
            all_train_feats, train_domain_labels, all_held_out_feats,
            heldout_domain_labels)

        feat_dim = all_feats.shape[-1]

        logging.info("Obtaining heads")
        callbacks = [
            skorch.callbacks.LRScheduler(
                torch.optim.lr_scheduler.StepLR,
                gamma=lr_decay_gamma,
                step_size=self._train_epochs / 2,
            ),
            skorch.callbacks.EpochScoring(
                self.accuracy_fn,
                lower_is_better=False,
                name='val_accuracy',
            ),
            skorch.callbacks.EpochScoring(
                self.accuracy_fn,
                lower_is_better=False,
                name='train_accuracy',
                on_train=True,
            ),
            skorch.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                threshold=0.0001,
                threshold_mode='rel',
                lower_is_better=False,
            )
        ]

        heads = self.get_heads(
            num_head_batches,
            feat_dim=feat_dim,
            criterion=nn.CrossEntropyLoss,
            num_labels=2,
            max_lr=max_lr,
            lr_sweep_factor=lr_sweep_factor,
            train_split=skorch.dataset.CVSplit(train_val_split),
            batch_size=self._algorithm.hparams['batch_size'],
            callbacks=callbacks)

        val_accuracies = []
        train_accuracies = []
        for hidx, h in enumerate(heads):
            logging.info("Fitting head %d/%d" % (hidx, len(heads)))
            train_val_idx = int(trainval_test_split * all_feats.shape[0])
            h.fit(all_feats[:train_val_idx, :], all_labels[:train_val_idx])
            val_accuracies.append([x['val_accuracy'] for x in h.history][-1])
            train_accuracies.append(
                max([x['train_accuracy'] for x in h.history]))

        best_model_idx = np.argmax(val_accuracies)
        best_gen_accuracy = (heads[best_model_idx].accuracy(
            all_feats[train_val_idx:, :], all_labels[train_val_idx:]))

        return (self.convert_domain_classifier_accuracy_to_divergence(
            best_gen_accuracy),
                self.convert_domain_classifier_accuracy_to_divergence(
                    max(train_accuracies)))

    def _calculate_lambda_closeness(self,
                                    all_train_feats,
                                    all_train_labels,
                                    all_held_out_feats,
                                    all_held_out_labels,
                                    lr_decay_gamma,
                                    num_head_batches,
                                    max_lr,
                                    lr_sweep_factor,
                                    train_env_to_use,
                                    train_val_split,
                                    trainval_test_split=0.8):

        # Permute the datapoints from train feats
        all_train_feats, all_train_labels = permute_dataset(
            all_train_feats, all_train_labels)
        all_held_out_feats, all_held_out_labels = permute_dataset(
            all_held_out_feats, all_held_out_labels)

        ndata = min(all_train_feats.shape[0], all_held_out_feats.shape[0])
        all_train_feats = all_train_feats[:ndata, :]
        all_train_labels = all_train_labels[:ndata]
        all_held_out_feats = all_held_out_feats[:ndata, :]
        all_held_out_labels = all_held_out_labels[:ndata]

        all_feats = torch.vstack([all_train_feats, all_held_out_feats])
        all_labels = torch.hstack([all_train_labels, all_held_out_labels])
        all_feats, all_labels = permute_dataset(all_feats, all_labels)

        feat_dim = all_train_feats.shape[-1]

        logging.info("Obtaining heads")
        callbacks = [
            skorch.callbacks.LRScheduler(
                torch.optim.lr_scheduler.StepLR,
                gamma=lr_decay_gamma,
                step_size=self._train_epochs / 2,
            ),
            skorch.callbacks.EpochScoring(
                self.accuracy_fn,
                lower_is_better=False,
                name='val_accuracy',
            ),
            skorch.callbacks.EpochScoring(
                self.accuracy_fn,
                lower_is_better=False,
                name='train_accuracy',
                on_train=True,
            ),
            skorch.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                threshold=0.0001,
                threshold_mode='rel',
                lower_is_better=False,
            )
        ]

        heads = self.get_heads(
            num_head_batches,
            feat_dim=feat_dim,
            criterion=nn.CrossEntropyLoss,
            num_labels=len(torch.unique(all_train_labels)),
            max_lr=max_lr,
            lr_sweep_factor=lr_sweep_factor,
            train_split=skorch.dataset.CVSplit(train_val_split),
            batch_size=self._algorithm.hparams['batch_size'],
            callbacks=callbacks)

        val_accuracies = []
        for hidx, h in enumerate(heads):
            logging.info("Fitting head %d/%d" % (hidx, len(heads)))
            train_val_idx = int(trainval_test_split * all_feats.shape[0])
            h.fit(all_feats[:train_val_idx, :], all_labels[:train_val_idx])
            val_accuracies.append([x['val_accuracy'] for x in h.history][-1])

        best_model_idx = np.argmax(val_accuracies)
        best_accuracy = heads[best_model_idx].accuracy(
            all_feats[train_val_idx:, :], all_labels[train_val_idx:])

        return 2.0 * (2.0 * best_accuracy - 1)

    def _calculate_measure_base(self,
                                num_head_batches=8,
                                max_lr=0.025,
                                lr_sweep_factor=0.5,
                                lr_decay_gamma=0.1,
                                stochastic_fraction_data=1.0,
                                train_env_to_use=None,
                                train_val_split=0.7):
        if train_env_to_use == None:
            train_env_to_use = self._train_loader

        featurizer = self._trainer_current.get_featurizer()
        _freeze_params(featurizer)

        logging.info("Precomputing features for train")
        all_train_feats, all_train_labels, _, _ = self.compute_features(
            train_env_to_use, featurizer, self._device)
        train_domain_labels = torch.ones_like(all_train_labels)

        logging.info("Precomputing features for held out")
        all_held_out_feats, all_held_out_labels, _, _ = self.compute_features(
            self._union_held_out_loader, featurizer, self._device)
        heldout_domain_labels = torch.zeros_like(all_held_out_labels)

        all_train_feats, all_train_labels = permute_dataset(
            all_train_feats, all_train_labels)
        data_idx = int(stochastic_fraction_data * all_train_feats.shape[0])
        all_train_feats = all_train_feats[:data_idx, :]
        all_train_labels = all_train_labels[:data_idx]

        all_held_out_feats, all_held_out_labels = permute_dataset(
            all_held_out_feats, all_held_out_labels)
        data_idx = int(stochastic_fraction_data * all_held_out_feats.shape[0])
        all_held_out_feats = all_held_out_feats[:data_idx, :]
        all_held_out_labels = all_held_out_labels[:data_idx]

        heldout_measure, train_measure = self._calculate_divergence_measure(
            all_train_feats, train_domain_labels, all_held_out_feats,
            heldout_domain_labels, lr_decay_gamma, num_head_batches, max_lr,
            lr_sweep_factor, train_env_to_use, train_val_split)

        # Compute lambda closenes
        lambda_closeness = self._calculate_lambda_closeness(
            all_train_feats, all_train_labels, all_held_out_feats,
            all_held_out_labels, lr_decay_gamma, num_head_batches, max_lr,
            lr_sweep_factor, train_env_to_use, train_val_split)

        return heldout_measure, train_measure, lambda_closeness

    def _calculate_measure(self,
                           *args,
                           stochastic_fraction_data=1.0,
                           **kwargs):

        if self._per_env == True:
            loader_list_to_use = self._train_loaders
        else:
            loader_list_to_use = [self._train_loader
                                  ]  # This is the union of all train loaders

        measures = []
        lambda_closeness_list = []
        for train_env in loader_list_to_use:
            held_out_measure, train_measure, lambda_closeness = self._calculate_measure_base(
                *args,
                **kwargs,
                train_env_to_use=train_env,
                stochastic_fraction_data=stochastic_fraction_data)
            if self._train_or_test == 'train':
                measures.append(train_measure)
            elif self._train_or_test == 'test':
                measures.append(held_out_measure)

            self.reset_held_out_loader()
            self.reset_train_loader()

            lambda_closeness_list.append(lambda_closeness)

        return np.sum(measures), {
            "lambda_closeness": np.sum(lambda_closeness_list)
        }


class HDelHDivergence(ClassifierTwoSampleTest):
    @staticmethod
    def hdh_accuracy_fn(net, X, y=None):
        dataloader = DataLoader(X, batch_size=256)
        score = 0
        total = 0
        training = net.module_.training

        net.module_.eval()
        for this_X, this_y in dataloader:
            this_y[this_y == 0] = -1.0
            net_one_logits, net_two_logits = net.module_.forward(this_X)
            net_one_preds = torch.argmax(net_one_logits, dim=-1).cpu()
            net_two_preds = torch.argmax(net_two_logits, dim=-1).cpu()
            score += float(((net_one_preds != net_two_preds).float() *
                            this_y.cpu()).sum())

            total += float((this_y == 1).cpu().sum())

        if training:
            net.module_.train()

        return 2.0 * float(score) / total

    def get_hdh_heads(self,
                      num_batch_heads,
                      feat_dim,
                      num_labels,
                      criterion,
                      batch_size=256,
                      max_lr=0.0025,
                      lr_sweep_factor=0.5,
                      weight_decay_max=1e1,
                      weight_decay_min=1e-4,
                      callbacks=None,
                      train_split=None):
        """Get heads for optimization.

        Set the maximum learning rate to start from for each head, and multiple
        heads then have a learning rate that is lr_sweep_factor^(i-1) * max_lr
        for the i'th head that we consider.
        """
        if callbacks == None:
            callbacks = []

        heads = []
        for hidx in range(num_batch_heads):
            lr = max_lr * lr_sweep_factor**(hidx)
            weight_decay = float(torch.multinomial(torch.logspace(1, -3, 5),
                                                   1))

            logging.info(
                f"Creating head {hidx} with lr {lr}, weight decay {weight_decay}"
            )
            if self._v_plus == True:
                first_head = nn.Sequential(
                    nn.Linear(feat_dim, int(feat_dim / 2)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 2), int(feat_dim / 4)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 4), int(feat_dim / 4)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 4), num_labels))
                second_head = nn.Sequential(
                    nn.Linear(feat_dim, int(feat_dim / 2)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 2), int(feat_dim / 4)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 4), int(feat_dim / 4)), nn.ReLU(),
                    nn.Linear(int(feat_dim / 4), num_labels))

            else:
                first_head = nn.Linear(feat_dim, num_labels)
                second_head = nn.Linear(feat_dim, num_labels)

            this_head = TrainEvalNeuralNet(
                module=ConcatModule(first_head, second_head),
                device=self._device,
                criterion=criterion,
                optimizer=torch.optim.SGD,
                optimizer__lr=lr,
                batch_size=batch_size,
                max_epochs=self._train_epochs,
                train_split=
                train_split,  # None means no validation, only training
                iterator_train__shuffle=True,
                callbacks=callbacks)

            heads.append(this_head)

        return heads

    def _calculate_divergence_measure(self,
                                      all_train_feats,
                                      train_domain_labels,
                                      all_held_out_feats,
                                      heldout_domain_labels,
                                      lr_decay_gamma,
                                      num_head_batches,
                                      max_lr,
                                      lr_sweep_factor,
                                      train_env_to_use,
                                      train_val_split,
                                      trainval_test_split=0.8):

        if (train_domain_labels - 1).sum() != 0:
            raise ValueError(
                "Train domain labels must be encoded with label 1")

        if (heldout_domain_labels).sum() != 0:
            raise ValueError(
                "Held out domain labels must be encoded with label 0")

        feat_dim = all_train_feats.shape[-1]
        all_train_feats, train_domain_labels = permute_dataset(
            all_train_feats, train_domain_labels)
        all_held_out_feats, heldout_domain_labels = permute_dataset(
            all_held_out_feats, heldout_domain_labels)

        num_data = min(all_train_feats.shape[0], all_held_out_feats.shape[0])
        all_train_feats = all_train_feats[:num_data, :]
        train_domain_labels = train_domain_labels[:num_data]
        all_held_out_feats = all_held_out_feats[:num_data, :]
        heldout_domain_labels = heldout_domain_labels[:num_data]

        all_feats = torch.vstack([all_train_feats, all_held_out_feats])
        all_labels = torch.hstack([train_domain_labels, heldout_domain_labels])
        all_feats, all_labels = permute_dataset(all_feats, all_labels)

        callbacks = [
            skorch.callbacks.LRScheduler(
                torch.optim.lr_scheduler.StepLR,
                gamma=lr_decay_gamma,
                step_size=self._train_epochs / 2,
            ),
            skorch.callbacks.EpochScoring(
                self.hdh_accuracy_fn,
                lower_is_better=False,
                name='val_divergence',
            ),
            skorch.callbacks.EpochScoring(
                self.hdh_accuracy_fn,
                lower_is_better=False,
                name='train_divergence',
                on_train=True,
            ),
            skorch.callbacks.EarlyStopping(
                monitor='val_divergence',
                patience=15,
                threshold=0.0001,
                threshold_mode='rel',
                lower_is_better=False,
            )
        ]

        heads = self.get_hdh_heads(
            num_head_batches,
            feat_dim=feat_dim,
            criterion=NegHDelHCriterion,
            num_labels=self._num_classes,
            max_lr=max_lr,
            lr_sweep_factor=lr_sweep_factor,
            train_split=skorch.dataset.CVSplit(train_val_split),
            batch_size=self._algorithm.hparams['batch_size'],
            callbacks=callbacks)

        val_divergence = []
        train_divergence = []
        for hidx, h in enumerate(heads):
            logging.info("Fitting head %d/%d" % (hidx, len(heads)))
            train_val_idx = int(trainval_test_split * all_feats.shape[0])
            h.fit(all_feats[:train_val_idx, :], all_labels[:train_val_idx])
            val_divergence.append([x['val_divergence'] for x in h.history][-1])
            train_divergence.append(
                max([x['train_divergence'] for x in h.history]))

        best_model_idx = np.argmax(val_divergence)
        h_del_h_divergence = self.hdh_accuracy_fn(
            heads[best_model_idx],
            torch.utils.data.TensorDataset(all_feats[train_val_idx:, :],
                                           all_labels[train_val_idx:]))

        return h_del_h_divergence, max(train_divergence)


class MMD(FunctionVMinimality):
    def __init__(self, *args, kernel_type="gaussian", **kwargs):
        super(MMD, self).__init__(*args, **kwargs)
        self._kernel_type = kernel_type

    def _calculate_measure(self):
        featurizer = self._trainer_current.get_featurizer()
        _freeze_params(featurizer)

        logging.info("Precomputing features for train")
        all_train_feats, _, _, _ = self.compute_features(
            self._train_loader, featurizer, self._device)

        logging.info("Precomputing features for held out")
        all_held_out_feats, _, _, _ = self.compute_features(
            self._union_held_out_loader, featurizer, self._device)

        return mmd(all_train_feats, all_held_out_feats, self._kernel_type), {}


class JacobianNorm(GenMeasure):
    def _jacobian_norm(self, algorithm, dataloader):
        def f(x):
            return softmax(algorithm.predict(x), dim=-1)

        all_jacobian_norms = []
        for idx, (x, _) in enumerate(dataloader):
            x = x.to(self._device)
            for this_x_idx in range(x.shape[0]):
                this_jacobian = torch.autograd.functional.jacobian(
                    f, x[this_x_idx].unsqueeze(0), strict=True)
                all_jacobian_norms.append(
                    torch.norm(this_jacobian.detach().cpu(), p='fro'))

            if (idx + 1) % 10 == 0:
                logging.info(f"Finished {idx+1}/{len(dataloader)}")

        logging.info("Computed Jacobian Norm")

        return all_jacobian_norms

    def _calculate_measure(self):
        all_gen_measure = self._jacobian_norm(self._algorithm,
                                              self._union_held_out_loader)
        return np.mean(all_gen_measure), {}


class JacobianNormRelative(JacobianNorm):
    def compute_train_test_norms(self):
        train_gen_measure = np.array(
            self._jacobian_norm(self._algorithm, self._train_loader))

        held_out_gen_measure = np.array(
            self._jacobian_norm(self._algorithm, self._union_held_out_loader))

        return train_gen_measure, held_out_gen_measure

    def _calculate_measure(self):
        train_gen_measure, held_out_gen_measure = self.compute_train_test_norms(
        )
        return np.mean(held_out_gen_measure) / np.mean(train_gen_measure), {}


class JacobianNormRelativeDiff(JacobianNormRelative):
    def _calculate_measure(self):
        train_gen_measure, held_out_gen_measure = self.compute_train_test_norms(
        )
        return np.mean(held_out_gen_measure) - np.mean(train_gen_measure), {}


class JacobianNormRelativeLogDiff(JacobianNormRelative):
    def _calculate_measure(self):
        train_gen_measure, held_out_gen_measure = self.compute_train_test_norms(
        )
        return np.mean(np.log(held_out_gen_measure + 1e-12)) - np.mean(
            np.log(train_gen_measure + 1e-12)), {}


class ValidationAccuracy(GenMeasure):
    def _calculate_measure(self, return_all=False):
        accuracy_list = []
        for loader in self._held_out_loader_list:
            accuracy_list.append(
                accuracy_fn(self._algorithm, loader, None, self._device))
        # Accuracy fn sets the algorithm to training, so undo it
        self._algorithm.eval()

        if return_all == True:
            return accuracy_list

        return np.mean(accuracy_list), {}


class Mixup(GenMeasure):
    def __init__(self, *args, alpha=None, **kwargs):
        if alpha is None:
            raise ValueError("Must specify alpha")
        super(Mixup, self).__init__(*args, **kwargs)
        self._alpha = alpha

    def _calculate_mixup(self, dataloader):
        errors = []
        for idx, (orig_images, _) in enumerate(dataloader):
            orig_images = orig_images.to(self._device)
            perm_batch_indices = torch.randperm(orig_images.shape[0]).long()
            perm_images = orig_images[perm_batch_indices]
            batch_lambdas = torch.Tensor(
                np.random.beta(self._alpha,
                               self._alpha,
                               size=orig_images.shape[0])).cuda()
            # First a forward pass with original images
            orig_f_x = softmax(self._algorithm.predict(orig_images), dim=1)

            # Next a forward pass with modified images
            perm_f_x = softmax(self._algorithm.predict(perm_images), dim=1)

            # Next a forward pass with linear combination of images
            expand_dims = orig_images.dim() - 1
            comb_f_x = softmax(self._algorithm.predict(
                orig_images * batch_lambdas.view(-1, *[1] *
                                                 (expand_dims)) + perm_images *
                (1.0 - batch_lambdas).view(-1, *[1] * (expand_dims))),
                               dim=1)

            this_error = (comb_f_x -
                          (orig_f_x * batch_lambdas.view(-1, 1) + perm_f_x *
                           (1.0 - batch_lambdas).view(-1, 1))).square().sum(-1)
            errors.extend(this_error.cpu().numpy())

            if (idx + 1) % 10 == 0:
                logging.info(f"Finished {idx+1}/{len(dataloader)}")

        logging.info("Computed Mixup")

        return np.mean(errors)

    def _calculate_measure(self, num_batch_computation=4):
        """Calculate the mixup generalization measure.

        Args:
          num_batch_computation: int, number of stochastic choices of \lambda
            to average over, when computing mixup
        Returns:
          Mixup generalization measure averaged across batches
        """
        mixup_gen_measures = []
        for _ in range(num_batch_computation):
            mixup_gen_measures.append(
                self._calculate_mixup(self._union_held_out_loader))
            self.reset_held_out_loader()
        return np.mean(mixup_gen_measures), {}


class MixupRelative(Mixup):
    def _calculate_measure_base(self, num_batch_computation=4):
        train_mixup_measures = []
        for _ in range(num_batch_computation):
            train_mixup_measures.append(
                self._calculate_mixup(self._train_loader))
            self.reset_train_loader()

        heldout_mixup_measures = []
        for _ in range(num_batch_computation):
            heldout_mixup_measures.append(
                self._calculate_mixup(self._union_held_out_loader))
            self.reset_held_out_loader()
        return np.array(train_mixup_measures), np.array(heldout_mixup_measures)

    def _calculate_measure(self, num_batch_computation=4):
        train_mixup_measures, heldout_mixup_measures = self._calculate_measure_base(
            num_batch_computation)
        return np.mean(heldout_mixup_measures) / np.mean(
            train_mixup_measures), {}


class MixupRelativeDiff(MixupRelative):
    def _calculate_measure(self, num_batch_computation=4):
        train_mixup_measures, heldout_mixup_measures = self._calculate_measure_base(
            num_batch_computation)
        return np.mean(heldout_mixup_measures) - np.mean(
            train_mixup_measures), {}


class MixupRelativeLogDiff(MixupRelative):
    def _calculate_measure(self, num_batch_computation=4):
        train_mixup_measures, heldout_mixup_measures = self._calculate_measure_base(
            num_batch_computation)
        return np.mean(np.log(heldout_mixup_measures + 1e-12)) - np.mean(
            np.log(train_mixup_measures + 1e-12)), {}

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Utilities for out of distribution generalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import copy
from skorch import NeuralNet

from itertools import zip_longest

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from domainbed.lib.fast_data_loader import _InfiniteSampler


# Code from Decodable IB paper, written by Yann Dubois
def get_device(module):
    """return device of module."""
    return next(module.parameters()).device


# Code from Decodable IB paper, written by Yann Dubois
class BatchNormConv(nn.Module):
    """Replace a batchnorm layer with a frozen convolution."""
    def __init__(self, batchnorm):
        super().__init__()
        if isinstance(batchnorm, nn.BatchNorm2d):
            conv = nn.Conv2d(
                batchnorm.num_features,
                batchnorm.num_features,
                1,
                groups=batchnorm.num_features,
            )
        elif isinstance(batchnorm, nn.BatchNorm1d):
            conv = nn.Conv1d(
                batchnorm.num_features,
                batchnorm.num_features,
                1,
                groups=batchnorm.num_features,
            )

        conv.eval()
        nn.init.ones_(conv.weight)
        nn.init.zeros_(conv.bias)
        conv.to(get_device(batchnorm))
        self.bn = nn.utils.fusion.fuse_conv_bn_eval(conv, batchnorm)

    def forward(self, x):
        return self.bn(x)


class NegCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return -1 * super(NegCrossEntropyLoss, self).forward(input, target)


class NeuralNetWithOptimInput(NeuralNet):
    """This class takes an object of the Optim class as input.
    
    This is in contrast to the regular skorch behavior, which
    takes the optimizer class as the input, and constructs
    the object in the initialize_optimizer routine.
    """
    def __init__(self, *args, **kwargs):
        super(NeuralNetWithOptimInput, self).__init__(*args, **kwargs)
        if len(self.get_params_for('module')) != 0:
            raise ValueError(
                "Must provide an initialized module as input, not parameters for modules"
            )

    def initialize_optimizer(self):
        self.optimizer_ = self.optimizer

    def initialize_except_module(self):
        self.initialize_virtual_params()
        self.initialize_callbacks()
        self.initialize_criterion()
        self.initialize_optimizer()
        self.initialize_history()

        self.initialized_ = True
        return self

    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError("Please use fit_partial only")

    def get_featurizer(self):
        if len(self.module_) != 2:
            raise ValueError("Expect featurizer and classifier")
        return self.module_[0]

    def get_classifier(self):
        if len(self.module_) != 2:
            raise ValueError("Expect featurizer and classifier")
        return self.module_[1]


def convert_algorithm_to_trainer(algorithm,
                                 device,
                                 criterion=nn.CrossEntropyLoss,
                                 featurizer_only=False):

    trainer = NeuralNetWithOptimInput(
        module=algorithm.network,
        criterion=criterion,
        optimizer=algorithm.optimizer,
        device=device,
    )

    return trainer


def zip_longest_padded(*iterables):
    # zip('ABCD', 'xy') --> Ax By Cx Dy
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    iterator_lengths = [len(it) for it in iterables]

    for _ in range(max(iterator_lengths)):
        result = []
        for it_idx, it in enumerate(iterators):
            elem = next(it, sentinel)
            if elem is sentinel:
                iterators[it_idx] = iter(iterables[it_idx])
                elem = next(iterators[it_idx], sentinel)

            result.append(elem)
        yield tuple(result)


class UnionDataLoader(object):
    """Generate minibatches from the union of a list of dataloaders"""
    def __init__(self, list_of_loaders, type='longest'):
        """Initialize a dataloader to provide samples from
        union of loaders.
        
        Args:
          list_of_loaders: list of `torch.utils.data.DataLoader` objects
          type: str, 'longest' or 'longest_padded' or 'shortest'
        """
        self._list_of_loaders = list_of_loaders
        self._type = type

        if type == 'longest':
            self._minibatches_iterator = zip_longest(*list_of_loaders)
        elif type == 'longest_padded':
            self._minibatches_iterator = zip_longest_padded(*list_of_loaders)
        elif type == 'shortest':
            self._minibatches_iterator = zip(*list_of_loaders)
        else:
            raise ValueError(f"Invalid type -- {type} provided.")

        self._all_cumulative_datapoints = list(
            np.cumsum([x.num_datapoints for x in list_of_loaders]))
        self._all_cumulative_datapoints.insert(0, 0)

        self._max_idx = self._all_cumulative_datapoints[-1]
        del self._all_cumulative_datapoints[-1]

    def __iter__(self):
        for _ in range(len(self)):
            xyidx_minibatches = next(self._minibatches_iterator)
            xyidx_minibatches = [
                xyidx for xyidx in xyidx_minibatches if xyidx is not None
            ]
            if len(xyidx_minibatches[0]) == 2:
                all_x = torch.cat([x for x, _ in xyidx_minibatches])
                all_y = torch.cat([y for _, y in xyidx_minibatches])

                yield all_x, all_y
            else:
                all_x = torch.cat([x for x, _, _ in xyidx_minibatches])
                all_y = torch.cat([y for _, y, _ in xyidx_minibatches])

                all_idx = []
                for env_idx, _ in enumerate(xyidx_minibatches):
                    all_idx.append(xyidx_minibatches[env_idx][2] +
                                   self._all_cumulative_datapoints[env_idx])
                all_idx = torch.cat(all_idx)

                yield all_x, all_y, all_idx

    def __len__(self):
        loader_sizes = [len(x) for x in self._list_of_loaders]

        if self._type == 'longest' or self._type == 'longest_padded':
            return max(loader_sizes)

        return min(loader_sizes)

    @property
    def max_idx(self):
        return self._max_idx


def _freeze_params(model):
    """Freeze parameters of an nn.Module"""
    for param in model.parameters():
        param.requires_grad = False

    return model


def _unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True

    return model


class _SplitDatasetAndReturnIndex(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys, include_index=False):
        super(_SplitDatasetAndReturnIndex, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.include_index = include_index

    def __getitem__(self, key):
        if self.include_index == True:
            return (*self.underlying_dataset[self.keys[key]], self.keys[key])
        else:
            return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0, include_index=False):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDatasetAndReturnIndex(
        dataset, keys_1,
        include_index), _SplitDatasetAndReturnIndex(dataset, keys_2,
                                                    include_index)


class DeterministicFastDataLoader(object):
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch, with deterministic example order."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            batch_size=batch_size,
            drop_last=False)

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler)))

        self._length = len(batch_sampler)
        self._batch_size = batch_size
        self._num_datapoints = self._batch_size * self._length

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    @property
    def num_datapoints(self):
        return self._num_datapoints

    def __len__(self):
        return self._length


def reset_weights_init(m):
    if 'reset_parameters' in dir(m):
        m.reset_parameters()


class ConcatModule(nn.Module):
    def __init__(self, module_one, module_two):
        super(ConcatModule, self).__init__()
        self._module_list = nn.ModuleList([module_one, module_two])

    def forward(self, x):
        outputs = []
        for m in self._module_list:
            outputs.append(m(x))
        return outputs


def permute_dataset(features, labels):
    perm = torch.randperm(features.shape[0])
    features = features[perm, :]
    labels = labels[perm]
    return features, labels


class TrainEvalNeuralNet(NeuralNet):
    def mean_train_loss(self, X, y):
        dtrain, _ = self.get_split_datasets(X, y)
        self.run_single_epoch(dtrain,
                              training=False,
                              prefix="train_eval",
                              step_fn=self.validation_step)
        losses = [
            x['train_eval_loss'] for x in self.history[-1]['batches']
            if 'train_eval_loss' in x.keys()
        ]
        return np.mean(losses)

    def accuracy(self, X, y, batch_size=256):
        """Computes accuracies on the provided datapoints.
        
        Args:
          X: `torch.Tensor`, input to the NeuralNet
          y: `torch.Tensor`, outputs from the NeuralNet, size [batch_size]
          batch_size: int, size of the batch
        Returns:
          accuracy: float, accuracy of the model on inputs X
        """
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        if y.dim() > 1:
            raise ValueError("Expect 1-D Tensor for labels")

        correct = 0
        total = 0
        training = self.module_.training

        self.module_.eval()
        for this_X, this_y in dataloader:
            correct += self.module_.forward(this_X).argmax(1).eq(
                this_y).sum().item()
            total += this_X.shape[0]

        if training:
            self.module_.train()

        return float(correct) / total


# Following code is from:
# https://github.com/facebookresearch/decodable_information_bottleneck
def add_noise_to_param_(module, sigma, is_relative=True):
    """Add uniform noise with standard deviation `sigma` to each weight."""
    with torch.no_grad():
        for param in module.parameters():
            if is_relative:
                unif = torch.distributions.uniform.Uniform(
                    0, torch.maximum(param.abs() * sigma, torch.tensor(1e-30)))
                noise = unif.sample()
            else:
                unif = torch.distributions.uniform.Uniform(0, sigma)
                noise = unif.sample(param.shape)
            param.add_(noise)


# Following code is from:
# https://github.com/facebookresearch/decodable_information_bottleneck
def clip_perturbated_param_(module,
                            unperturbated_params,
                            clip_factor,
                            is_relative=True):
    """
    Element wise clipping of the absolute value of the difference in weight. `unperturbated_params` 
    needs to be a dictionary of unperturbated param. Use `is_relative` if clip factor should multipy
    the unperturbated param.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            w = unperturbated_params[name]

            # delta_i = (delta+w)_i - w_i
            delta = param - w

            max_abs_delta = clip_factor * w.abs(
            ) if is_relative else clip_factor

            clipped_delta = torch.where(delta.abs() > max_abs_delta,
                                        delta.sign() * max_abs_delta, delta)

            # inplace replace
            param.fill_(0)
            param.add_(w)
            param.add_(clipped_delta)


# Following code is from:
# https://github.com/facebookresearch/decodable_information_bottleneck
def clone_trainer(trainer, is_reinit_besides_param=False):
    """Clone a trainer with optional possibility of reinitializing everything besides 
    parameters (e.g. optimizers.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer_new = copy.deepcopy(trainer)

    if is_reinit_besides_param:
        trainer_new.initialize_callbacks()
        trainer_new.initialize_criterion()
        trainer_new.initialize_optimizer()
        trainer_new.initialize_history()

    return trainer_new


# Following code is from:
# https://github.com/facebookresearch/decodable_information_bottleneck
def batchnorms2convs_(module):
    """Converts all the batchnorms to frozen convolutions."""
    for name, m in module.named_children():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            module._modules[name] = BatchNormConv(m)
        else:
            batchnorms2convs_(module._modules[name])


# Following code is from:
# https://github.com/facebookresearch/decodable_information_bottleneck
def get_exponential_decay_gamma(scheduling_factor, max_epochs):
    """Return the exponential learning rate factor gamma.
    Parameters
    ----------
    scheduling_factor :
        By how much to reduce learning rate during training.
    max_epochs : int
        Maximum number of epochs.
    """
    return (1 / scheduling_factor)**(1 / max_epochs)


# Following code is from:
# https://github.com/facebookresearch/decodable_information_bottleneck
class BaseRepresentation:
    """Compute the base representation for a number in a certain base while memoizing."""
    def __init__(self, base):
        self.base = base
        self.memoize = {0: []}

    def __call__(self, number):
        """Return a list of the base representation of number."""
        if number in self.memoize:
            return self.memoize[number]

        self.memoize[number] = self(number // self.base) + [number % self.base]
        return self.memoize[number]

    def get_ith_digit(self, number, i):
        """Return the ith digit pf the base representation of number."""
        digits = self(number)
        if i >= len(digits):
            return 0  # implicit padding with zeroes
        return digits[-i - 1]


# Following code is from:
# https://github.com/facebookresearch/decodable_information_bottleneck
class BaseRepIthDigits:
    """Compute the ith digit in a given base for torch batch of numbers while memoizing (in numpy)."""
    def __init__(self, base):
        base_rep = BaseRepresentation(base)
        self.base_rep = np.vectorize(base_rep.get_ith_digit)

    def __call__(self, tensor, i_digit):
        return self.base_rep(tensor, i_digit)
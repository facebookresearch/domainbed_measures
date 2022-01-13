"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Model specification loading and interaction with DomainBed Models.
"""
import torch
import random
import numpy as np
import logging

from collections import namedtuple
from typing import Dict

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc
from domainbed_measures.utils import DeterministicFastDataLoader
from domainbed_measures.utils import split_dataset


def load_from_state(args: Dict,
                    hparams: Dict,
                    algorithm_dict: Dict,
                    dirty_ood_split: str,
                    target_test_env: int,
                    include_index=False):
    """Load from the model checkpoints things like weights and dataloaders.

    Args:
      args: Args used to train the model
      hparams: Hyperparameters used for the model run
      algorithm_dict: Weights of the trained model
      target_test_env: Test environment we want to load (among a list of
        competing alternatives)
      include_index: Whether to include the index of a datapoint along
        with label
    """
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if target_test_env not in args.test_envs:
        raise ValueError("Target test environment must be in "
                         "list of test envs used for model.")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in datasets.DATASETS:
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs,
                                               hparams)
    else:
        raise NotImplementedError

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = split_dataset(env,
                                 int(len(env) * args.holdout_fraction),
                                 misc.seed_hash(args.trial_seed, env_i),
                                 include_index=include_index)
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    train_loaders = [
        DeterministicFastDataLoader(dataset=env,
                                    batch_size=hparams['batch_size'],
                                    num_workers=dataset.N_WORKERS)
        for i, (env, _) in enumerate(in_splits) if i not in args.test_envs
    ]

    eval_loaders = [
        DeterministicFastDataLoader(dataset=env,
                                    batch_size=9,
                                    num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)
    ]

    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += [
        'env{}_out'.format(i) for i in range(len(out_splits))
    ]

    wd_eval_loader_names = [
        'env{}_out'.format(i) for i, _ in enumerate(in_splits)
        if i not in args.test_envs
    ]

    # Dirty OOD is the split we touch for computing measures
    dirty_ood_eval_loader_names = [
        'env{}_{}'.format(i, dirty_ood_split) for i, _ in enumerate(in_splits)
        if i == target_test_env
    ]

    clean_ood_split = 'in'
    if dirty_ood_split == 'in':
        clean_ood_split = 'out'

    clean_ood_eval_loader_names = [
        'env{}_{}'.format(i, clean_ood_split) for i, _ in enumerate(in_splits)
        if i == target_test_env
    ]

    train_loader_names = [
        'env{}_in'.format(i) for i, _ in enumerate(in_splits)
        if i not in args.test_envs
    ]

    logging.info("WD eval loaders:")
    logging.info(wd_eval_loader_names)

    logging.info("Dirty OOD eval loaders:")
    logging.info(dirty_ood_eval_loader_names)

    logging.info("Clean OOD eval loaders:")
    logging.info(clean_ood_eval_loader_names)

    logging.info("Train loaders:")
    logging.info(train_loader_names)

    wd_eval_loaders = []
    dirty_ood_eval_loaders = []
    clean_ood_eval_loaders = []

    for this_name, this_loader in zip(eval_loader_names, eval_loaders):
        if this_name in wd_eval_loader_names:
            wd_eval_loaders.append(this_loader)
        elif this_name in dirty_ood_eval_loader_names:
            dirty_ood_eval_loaders.append(this_loader)
        elif this_name in clean_ood_eval_loader_names:
            clean_ood_eval_loaders.append(this_loader)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict, strict=True)

    return (algorithm, train_loaders, wd_eval_loaders, dirty_ood_eval_loaders,
            clean_ood_eval_loaders, dataset.num_classes)


def load_model_and_dataloaders(model_pickle_path,
                               dirty_ood_split,
                               test_env_idx,
                               include_index=False):
    save_dict = torch.load(model_pickle_path)
    args = namedtuple('args', save_dict["args"].keys())(**save_dict["args"])
    hparams = save_dict["model_hparams"]
    algorithm_dict = save_dict["model_dict"]

    logging.info('Loaded HParams:')
    for k, v in sorted(hparams.items()):
        logging.info('\t{}: {}'.format(k, v))

    return load_from_state(args,
                           hparams,
                           algorithm_dict,
                           dirty_ood_split,
                           target_test_env=test_env_idx,
                           include_index=include_index)
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import pandas as pd
import os
import jsonlines
import logging
import jsonlines

from collections import defaultdict
from glob import glob
from filelock import FileLock
from submitit.helpers import Checkpointable
from typing import List

from domainbed_measures.measures.registry import MeasureRegistry
from domainbed_measures.measures.registry import _MNIST_NUM_EXAMPLES_FISHER
from domainbed_measures.model_spec import load_model_and_dataloaders

_OUT_FILE_NAME="results.jsonl"
_MODEL_FILE_NAME="model.pkl"
MODEL_SELECTION="latest"


def compute_performance_on_splits(algorithm, train_loaders, wd_loaders,
                                  ood_loaders, num_classes):
    ood_accuracy_class = MeasureRegistry()["validation_accuracy"](
        algorithm, None, ood_loaders, num_classes,
        **MeasureRegistry._KWARGS["validation_accuracy"]["measure_args"])
    ood_accuracy, _ = ood_accuracy_class.compute()

    train_accuracy_class = MeasureRegistry()["validation_accuracy"](
        algorithm, None, train_loaders, num_classes,
        **MeasureRegistry._KWARGS["validation_accuracy"]["measure_args"])
    train_accuracy, _ = train_accuracy_class.compute()

    wd_accuracy_class = MeasureRegistry()["validation_accuracy"](
        algorithm, None, wd_loaders, num_classes,
        **MeasureRegistry._KWARGS["validation_accuracy"]["measure_args"])
    wd_accuracy, _ = wd_accuracy_class.compute()

    ood_gen_gap = train_accuracy - ood_accuracy
    wd_gen_gap = train_accuracy - wd_accuracy

    return ood_gen_gap, wd_gen_gap, train_accuracy, ood_accuracy, wd_accuracy


def read_jsonl_result(out_filepath):
    results = pd.DataFrame()
    all_test_envs = None

    with jsonlines.open(out_filepath) as reader:
        for obj in reader:
            if all_test_envs is None:
                all_test_envs = obj["args"]["test_envs"]

            del obj["args"]
            del obj["hparams"]

            results = results.append(obj, ignore_index=True)

    return results, all_test_envs


def load_generalization_gap(out_results: pd.DataFrame,
                            test_envs: List[int],
                            test_env_idx: int,
                            dirty_ood_split: str,
                            model_selection: str = "latest") -> List[List]:
    # For out-of-domain accuracy only look at in+out performance
    if dirty_ood_split not in ["in", "out"]:
        raise ValueError(
            f"Invalid value for dirty_ood_split: {dirty_ood_split}")

    if model_selection != "latest":
        raise ValueError

    ood_out_domains = []
    # Columns with results are like 'env1_out_acc' or 'env2_in_acc' and so on.
    all_envs_acc = [
        x for x in out_results.columns if 'env' in x and 'acc' in x
    ]

    ood_out_domains = []
    wd_out_domains = []
    in_domains = []

    clean_ood_split = "in"
    if dirty_ood_split == "in":
        clean_ood_split = "out"
    del dirty_ood_split

    for e in all_envs_acc:
        if not ('in' in e or 'out' in e):
            raise ValueError("Unexpected env accuracy specifier %s" % (e))
        if int(e.split('_')[0].strip(
                'env')) == test_env_idx and clean_ood_split in e:
            ood_out_domains.append(e)
        elif int(e.split('_')[0].strip('env')) not in test_envs and 'out' in e:
            wd_out_domains.append(e)
        elif int(e.split('_')[0].strip('env')) not in test_envs and 'in' in e:
            in_domains.append(e)

    in_domain_perf = out_results[in_domains].mean(1)
    ood_out_domain_perf = out_results[ood_out_domains].mean(1)
    wd_out_domain_perf = out_results[wd_out_domains].mean(1)

    ood_gap = in_domain_perf - ood_out_domain_perf
    wd_gap = in_domain_perf - wd_out_domain_perf

    return (ood_gap.iloc[-1], wd_gap.iloc[-1], in_domain_perf.iloc[-1],
            ood_out_domain_perf.iloc[-1], wd_out_domain_perf.iloc[-1])


class ExperimentBase(Checkpointable):
    def __init__(self, dirty_ood_split):
        self._dirty_ood_split = dirty_ood_split

    def __call__(self, path, measure, dataset, test_env_idx):

        results = []
        ######## Compute the generalization measure #####################
        # Need to reload the data since the config of the dataloader also depends on which
        # measure we use.
        algorithm, train_loaders, wd_eval_loaders, dirty_ood_eval_loaders, clean_ood_eval_loaders, num_classes = (
            load_model_and_dataloaders(
                os.path.join(path, _MODEL_FILE_NAME), self._dirty_ood_split,
                test_env_idx, **MeasureRegistry._KWARGS[measure]["data_args"]))

        logging.info("Computing WD generalization gap")
        MeasureClass = MeasureRegistry()[measure]

        # Optional file to store temporary results and runs at for caching
        if 'fisher' in measure:
            if "MNIST" in dataset:
                logging.info(
                    f"Increasing number of examples for MNIST to {_MNIST_NUM_EXAMPLES_FISHER}."
                )
                MeasureRegistry._KWARGS[measure]["measure_args"][
                    "max_num_examples"] = _MNIST_NUM_EXAMPLES_FISHER

        measure_class_wd = MeasureClass(
            algorithm,
            train_loaders,
            wd_eval_loaders,
            num_classes,
            **MeasureRegistry._KWARGS[measure]["measure_args"])
        gen_measure_val_wd, metadata_wd = measure_class_wd.compute()

        # We only need to compute the generalization measure for out of distribution
        # if the measure uses out of distribution data, if not we need not compute
        if MeasureRegistry._KWARGS[measure]["measure_args"].get(
                'use_eval_data') == True:
            logging.info("Computing OOD generalization gap")

            measure_class_ood = MeasureClass(
                algorithm,
                train_loaders,
                dirty_ood_eval_loaders,
                num_classes,
                **MeasureRegistry._KWARGS[measure]["measure_args"])
            gen_measure_val_ood, metadata_ood = measure_class_ood.compute()
        else:
            gen_measure_val_ood, metadata_ood = gen_measure_val_wd, metadata_wd

        return {
            "gen_measure_val_wd": float(gen_measure_val_wd),
            "gen_measure_val_ood": float(gen_measure_val_ood),
            "metadata_wd": metadata_wd,
            "metadata_ood": metadata_ood,
            "measure": measure,
            "dataset": dataset,
            "path": path,
            "test_env": test_env_idx,
        }


class Experiment(ExperimentBase):
    @staticmethod
    def sanity_check(out_results, test_envs, test_env_idx, dirty_ood_split,
                     model_selection, path):
        logging.info(
            "Sanity check on accuracy match between loaded and stored data.")
        _, _, chk_in_domain_perf, chk_ood_out_domain_perf, chk_wd_out_domain_perf = (
            load_generalization_gap(out_results,
                                    test_envs,
                                    test_env_idx,
                                    dirty_ood_split,
                                    model_selection=model_selection))

        algorithm, train_loaders, wd_eval_loaders, dirty_ood_eval_loaders, clean_ood_eval_loaders, num_classes = (
            load_model_and_dataloaders(
                os.path.join(path,
                             _MODEL_FILE_NAME), dirty_ood_split, test_env_idx,
                **MeasureRegistry._KWARGS["validation_accuracy"]["data_args"]))

        ######## Compute the validation accuracy to verify #############
        ood_gen_gap, wd_gen_gap, in_domain_perf, ood_out_domain_perf, wd_out_domain_perf = (
            compute_performance_on_splits(algorithm, train_loaders,
                                          wd_eval_loaders,
                                          clean_ood_eval_loaders, num_classes))
        del clean_ood_eval_loaders

        if chk_in_domain_perf != in_domain_perf:
            logging.warning(f"Mismatch between loaded-in-domain-performance "
                            f"{chk_in_domain_perf} and computed-in-domain"
                            f"-performance {in_domain_perf}")
        return ood_gen_gap, wd_gen_gap, in_domain_perf, ood_out_domain_perf, wd_out_domain_perf

    @staticmethod
    def write_results(results, save_path):
        logging.info(f"Writing results to {save_path}.")
        if save_path is not None:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, 'results.jsonl')
            locker = FileLock(save_file + ".lock")
            with locker.acquire(timeout=60):
                with jsonlines.open(save_file, 'a') as writer:
                    writer.write(results)
                    locker.release()
        logging.info(f"Finished writing results to {save_path}.")

    def __call__(self,
                 path: str,
                 measure_or_measure_list: list,
                 dataset: str,
                 save_path=None):
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        measures = measure_or_measure_list

        if not isinstance(measure_or_measure_list, list) and (isinstance(
                measure_or_measure_list, str)):
            measures = [measure_or_measure_list]
        elif not isinstance(measure_or_measure_list, list):
            raise ValueError("Unexpected type for measure_or_measure_list")
        np.random.shuffle(measures)

        out_results, test_envs = read_jsonl_result(
            os.path.join(path, _OUT_FILE_NAME))

        for test_env_idx in test_envs:
            if not isinstance(test_env_idx, int):
                raise ValueError("Expect an integer test environment id.")

            # Check if loaded and computed results match up
            ood_gen_gap, wd_gen_gap, in_domain_perf, ood_out_domain_perf, wd_out_domain_perf = (
                self.sanity_check(out_results, test_envs, test_env_idx,
                                  self._dirty_ood_split, MODEL_SELECTION,
                                  path))
            for idx, m in enumerate(measures):
                if m not in MeasureRegistry._VALID_MEASURES:
                    raise ValueError("Invalid measure.")

                logging.info(
                    f"Computing measure {m} for {path}, test_env {test_env_idx} -- ({idx + 1}/{len(measures)})"
                )
                results = super(Experiment,
                                self).__call__(path, m, dataset, test_env_idx)
                results["ood_gen_gap"] = ood_gen_gap
                results["wd_gen_gap"] = wd_gen_gap
                results["in_domain_perf"] = in_domain_perf
                results["ood_out_domain_perf"] = ood_out_domain_perf
                results["wd_out_domain_perf"] = wd_out_domain_perf

                logging.info(f"Finished measure {m} for {path}")
                self.write_results(results, save_path)


class VarianceExperiment(ExperimentBase):
    """Class to compute the variance of a measure."""
    def __call__(self,
                 path: str,
                 measure_or_measure_list: list,
                 dataset: str,
                 save_path=None,
                 num_trials=10):
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        measures = measure_or_measure_list

        if not isinstance(measure_or_measure_list, list) and (isinstance(
                measure_or_measure_list, str)):
            measures = [measure_or_measure_list]
        elif not isinstance(measure_or_measure_list, list):
            raise ValueError("Unexpected type for measure_or_measure_list")
        np.random.shuffle(measures)

        if not all(['c2st' in m or 'hdh' in m for m in measures]):
            raise NotImplementedError("Variance Experiment only implemented"
                                      "for c2st and hdh based measures.")

        out_results, test_envs = read_jsonl_result(
            os.path.join(path, _OUT_FILE_NAME))

        for test_env_idx in test_envs:
            if not isinstance(test_env_idx, int):
                raise ValueError("Expect an integer test environment id.")

            gen_measure_vals = defaultdict(list)

            for idx, m in enumerate(measures):
                if m not in MeasureRegistry._VALID_MEASURES:
                    raise ValueError("Invalid measure.")

                algorithm, train_loaders, wd_eval_loaders, dirty_ood_eval_loaders, clean_ood_eval_loaders, num_classes = (
                    load_model_and_dataloaders(
                        os.path.join(path, _MODEL_FILE_NAME),
                        self._dirty_ood_split, test_env_idx,
                        **MeasureRegistry._KWARGS[m]["data_args"]))
                MeasureClass = MeasureRegistry()[m]
                measure_class_ood = MeasureClass(
                    algorithm,
                    train_loaders,
                    dirty_ood_eval_loaders,
                    num_classes,
                    **MeasureRegistry._KWARGS[m]["measure_args"])

                for trial in range(num_trials):
                    gen_measure_val_ood, metadata_ood = measure_class_ood.compute(
                        stochastic_fraction_data=0.8)
                    gen_measure_vals[m].append(gen_measure_val_ood)

            self.write_results(gen_measure_vals, save_path)

    @staticmethod
    def write_results(results, save_path):
        logging.info(f"Writing results to {save_path}.")
        if save_path is not None:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, 'variance_results.jsonl')
            locker = FileLock(save_file + ".lock")
            with locker.acquire(timeout=60):
                with jsonlines.open(save_file, 'a') as writer:
                    writer.write(results)
                    locker.release()
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import pandas as pd
import glob
import json
import numpy as np
import ast

MAX_VAL=1e30
DATASETS_TO_NUM_ENVS = {
    "OfficeHome": 4,
    "VLCS": 4,
    "PACS": 4,
    "RotatedMNIST": 6,
}


def numeric_list_to_str(x):
    x = sorted(x)
    return '_' + '_'.join([str(ex) for ex in x])


def fix_hdh_c2st_divergence_from_sum_to_mean(hdh_or_c2st_divergence,
                                             dataset,
                                             num_env_test,
                                             is_hdh,
                                             per_env=True):
    # Makes changes to the originally computed hdh divergence as:
    # 2 * max_{h, h'} E_{source} p[h'(x) != h(x)] - E_{target} p[h'(x) != h(x)]
    #
    # for use in the generalization bounds, where we have a factor of 1/2 and we have
    # mean of hdh divergences across different source and train envs when we are in
    # the multi-source setting
    if per_env == True:
        num_env_train = DATASETS_TO_NUM_ENVS.get(dataset) - num_env_test
    else:
        num_env_train = 1
    if is_hdh:
        scaling_factor = 0.5
    else:
        scaling_factor = 1.0
    new_hdh_or_c2st_divergence = scaling_factor * hdh_or_c2st_divergence / float(
        num_env_train)
    return new_hdh_or_c2st_divergence


def fix_lambda_closeness(wrong_lambda_closeness_sum,
                         dataset,
                         num_env_test,
                         per_env=True):
    if dataset not in DATASETS_TO_NUM_ENVS.keys():
        raise ValueError(
            "Add details of number of environments to DATASETS_TO_NUM_ENVS")
    if per_env == True:
        num_env_train = DATASETS_TO_NUM_ENVS.get(dataset) - num_env_test
    else:
        num_env_train = 1

    env_sum_accuracy = (wrong_lambda_closeness_sum + num_env_train * 2.0) / 4.0
    # 2 is multiplied to get sum of source and target errors as opposed to
    # average of them (for each environment)
    env_sum_error_sum = 2 * (float(num_env_train) - env_sum_accuracy)

    return (env_sum_error_sum) / float(num_env_train)


def load_results(run_dir, experiment_path):
    """Load results from gen-ood runs.

    Args:
        run_dir (str): Base directory where we do all the runs and store results
          for a given domainbed run
        experiment_path (str): Format is _{in/out}_{algorithm}_{dataset}, name of
          the particular dataset and run we want to analyze

    Returns:
        pd.DataFrame: A dataframe object with the results
    """
    result_file_regex = os.path.join(run_dir, experiment_path, "slurm_files",
                                     "*/results.jsonl")
    result_files = glob.glob(result_file_regex)
    all_results = []

    for rfidx, rf in enumerate(result_files):
        with open(rf, 'r') as file:
            for line_idx, line in enumerate(file):
                try:
                    res = json.loads(line)
                except:
                    print(
                        f"Warning: Invalid line {line_idx} in {rf}. Skipping.."
                    )
                    continue
                if res['gen_measure_val_ood'] > MAX_VAL:
                    print(f"Skpping value greater than {MAX_VAL}")
                    continue
                with open(os.path.join(res['path'], 'out.txt')) as f:
                    test_env_str = f.readlines()[22]
                    if 'test_envs: ' not in test_env_str:
                        raise ValueError("Reading wrong line for test envs")
                    all_test_envs = sorted(
                        ast.literal_eval(test_env_str.lstrip().rstrip().lstrip(
                            "test_envs: ")))
                lambda_wd = res['metadata_wd'].get('lambda_closeness')
                lambda_ood = res['metadata_ood'].get('lambda_closeness')
                lambda_wd = fix_lambda_closeness(
                    lambda_wd, res['dataset'], len(all_test_envs), "per_env"
                    in res["measure"]) if lambda_wd is not None else -1

                lambda_ood = fix_lambda_closeness(
                    lambda_ood, res['dataset'], len(all_test_envs), "per_env"
                    in res["measure"]) if lambda_ood is not None else -1

                if 'hdh' in res['measure'] or 'c2st' in res['measure']:
                    res['gen_measure_val_ood'] = fix_hdh_c2st_divergence_from_sum_to_mean(
                        res['gen_measure_val_ood'], res['dataset'],
                        len(all_test_envs), "hdh" in res['measure'], "per_env"
                        in res["measure"])
                    res['gen_measure_val_wd'] = fix_hdh_c2st_divergence_from_sum_to_mean(
                        res['gen_measure_val_wd'], res['dataset'],
                        len(all_test_envs), "hdh" in res['measure'], "per_env"
                        in res["measure"])

                res['all_test_envs'] = all_test_envs
                res['lambda_wd'] = lambda_wd
                res['lambda_ood'] = lambda_ood
                res['path_and_test_envs'] = "%s_%s" % (
                    res['path'], str(res['all_test_envs']))
                all_results.append(res)
        print(f"{rfidx}/{len(result_files)} done \r", end="\r")

    all_results = pd.DataFrame(all_results)
    all_results = all_results.replace([np.inf, -np.inf], np.nan)
    all_results = all_results.dropna(axis=0)

    print(f"Jobs per measure:")
    for m in sorted(list(all_results['measure'].unique())):
        print(
            "%s: %d/%d" %
            (m, len(all_results[all_results['measure'] == m]['path'].unique()),
             len(all_results['path'].unique())))

    return all_results, sorted(list(all_results['measure'].unique()))

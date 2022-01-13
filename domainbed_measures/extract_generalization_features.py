"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Some tools for analyzing results and plotting.
"""
import argparse
import csv
import os
import pandas as pd


from tqdm import tqdm
from collections import defaultdict

from domainbed_measures.experiment.io_utils import load_results

parser = argparse.ArgumentParser()

parser.add_argument("--run_dir",
                    default="",
                    help="Directory with the runs and results.")

parser.add_argument("--sweep_name",
                    default="_out_ERM_VLCS",
                    help="Directory with the runs and results.")

parser.add_argument("--canonicalize",
                    action="store_true",
                    default=False,
                    help="whther to canonicalize")

args = parser.parse_args()

# Canonicalization is with respect to gap as the metric. if accuracy is the metric no canonicalization.
CANONICALIZATION = {
    "sharp_mag": 1,
    "entropy": -1,
    "entropy_held_out": -1,
    "path_norm": 1,
    "jacobian_norm": 1,
    "v_minimality": -1,
    "v_plus_minimality": -1,
    "mixup": +1,
    "mixup_relative": +1,
    "jacobian_norm_relative": 1,
    "jacobian_norm_relative_log_diff": 1,
    "jacobian_norm_relative_diff": 1,
    "fisher_eigval_sum_diff_ex_75": 1,
    "mixup_alpha_0.3": +1,
    "mixup_alpha_0.3_relative": +1,
    "fisher_eigval_sum_ex_75": 1,  # Should be called eigval sum ratio!!
    "fisher_eigvec_align_ex_75": -1,
    "c2st": 1,
    "v_plus_c2st": 1,
    "c2st_train": 1,
    "c2st_train_per_env": 1,
    "mmd_gaussian": 1,
    "mmd_mean_cov": 1,
    "mixup_relative_log_diff": +1,
    "mixup_relative_diff": +1,
    "wd_out_domain_err": -1,
}


def extract_features_for_regression(results,
                                    feature_name_list,
                                    wd_or_ood='ood',):
    def numeric_list_to_str(x):
        if isinstance(x, list):
            x = sorted(x)
            return '_' + '_'.join([str(ex) for ex in x])
        return x

    results['all_test_envs'] = results['all_test_envs'].apply(
        numeric_list_to_str)
    all_test_envs = results['all_test_envs'].unique()

    header = ["dataset", "all_test_envs", "test_env"]
    for f in feature_name_list:
        header.append(f)
        if "c2st" in f or "hdh" in f:
            header += ["%s_lambda" % (f)]
            header += ["%s_perr" % (f)]
            header += ["%s_perr_plambda" % (f)]
    header.append("target_err")
    if args.canonicalize == True:
        canonicalizer = CANONICALIZATION
    else:
        canonicalizer = defaultdict(lambda: 1)

    # We solve one regression problem per test environment
    all_rows = []
    for env in all_test_envs:
        print(f'Extracting features for env {env}')
        test_envs_with_results = list(
            results[results['all_test_envs'] == env]['test_env'].unique())

        for this_test_env in test_envs_with_results:
            results_env_test = results[(results['all_test_envs'] == env) &
                                       (results['test_env'] == this_test_env)]

            for _, path in tqdm(enumerate(results_env_test['path'].unique())):
                this_row = []
                this_row.append(
                    results_env_test[results_env_test['path'] == path]
                    ['dataset'].reset_index(drop=True)[0])
                this_row.append(env)
                this_row.append(this_test_env)

                feature = []
                for feat_name in feature_name_list:
                    feature_access_str = 'gen_measure_val_%s' % (wd_or_ood)
                    if feat_name in list(results_env_test[
                            results_env_test['path'] == path]['measure']):
                        feature.append(results_env_test[
                            (results_env_test['path'] == path)
                            & (results_env_test['measure'] == feat_name)]
                                       [feature_access_str].mean() *
                                       canonicalizer[feat_name])
                        if "c2st" in feat_name or "hdh" in feat_name:
                            c2st_or_hdh_feature = results_env_test[
                                (results_env_test['path'] == path)
                                & (results_env_test['measure'] == feat_name
                                   )][feature_access_str].mean()
                            lambda_closeness = results_env_test[
                                (results_env_test['path'] == path)
                                & (results_env_test['measure'] == feat_name)][
                                    "lambda_%s" % (wd_or_ood)].mean()
                            wd_err_path = (1.0 - results_env_test[
                                results_env_test['path'] == path]
                                           ['wd_out_domain_perf'].mean())
                            feature.append(lambda_closeness)
                            feature.append(c2st_or_hdh_feature + wd_err_path)
                            feature.append(c2st_or_hdh_feature + wd_err_path +
                                           lambda_closeness)
                    elif feat_name == 'wd_out_domain_err':
                        feature.append(
                            1.0 -
                            results_env_test[results_env_test['path'] == path]
                            ['wd_out_domain_perf'].mean() *
                            canonicalizer['wd_out_domain_err'])
                    else:
                        # Perform imputation based on the mean
                        feature.append(results_env_test[
                            results_env_test['measure'] == feat_name]
                                       [feature_access_str].mean() *
                                       canonicalizer[feat_name])
                        if pd.isnull(feature[-1]):
                            feature[-1] = 0.0

                        if "c2st" in feat_name or "hdh" in feat_name:
                            lambda_closeness = results_env_test[
                                results_env_test['measure'] == feat_name][
                                    "lambda_%s" % (wd_or_ood)].mean()

                            c2st_or_hdh_feature = results_env_test[(
                                results_env_test['measure'] == feat_name
                            )][feature_access_str].mean()
                            wd_err_path = (1.0 - results_env_test[
                                results_env_test['path'] == path]
                                           ['wd_out_domain_perf'].mean())

                            feature.append(lambda_closeness)
                            if pd.isnull(feature[-1]):
                                feature[-1] = -1.0
                            feature.append(c2st_or_hdh_feature + wd_err_path)
                            if pd.isnull(feature[-1]):
                                feature[-1] = -1.0
                            feature.append(c2st_or_hdh_feature + wd_err_path +
                                           lambda_closeness)
                            if pd.isnull(feature[-1]):
                                feature[-1] = -1.0

                this_row.extend(feature)
                if wd_or_ood == 'ood':
                    this_row.append(
                        1.0 -
                        results_env_test[results_env_test['path'] ==
                                         path]['ood_out_domain_perf'].mean())
                elif wd_or_ood == 'wd':
                    this_row.append(
                        1.0 -
                        results_env_test[results_env_test['path'] ==
                                         path]['wd_out_domain_perf'].mean())

                all_rows.append(this_row)

    return header, all_rows


def experiment_per_dataset(results,
                           feature_names,
                           experiment_path,
                           wd_or_ood='ood'):
    """[summary]

    Args:
        results ([type]): [description]
        feature_names ([type]): [description]
        run_dir ([type]): [description]
        experiment_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    print(f"Running analysis for dataset {experiment_path}")
    header, all_rows = extract_features_for_regression(results,
                                                       feature_names,
                                                       wd_or_ood=wd_or_ood)
    return header, all_rows


def analyze_results(run_dir, datasets):
    experiment_path_list = datasets.split(",")
    print("Running extraction on following datasets:")
    print(experiment_path_list)

    common_feature_names = None
    all_results = {}
    for experiment_path in experiment_path_list:
        results, feature_names = load_results(run_dir, experiment_path)
        all_results[experiment_path] = results

        if common_feature_names == None:
            common_feature_names = set(feature_names)
        else:
            common_feature_names = common_feature_names.intersection(
                set(feature_names))

    if common_feature_names == None:
        raise ValueError
    # Add within domain val accuracy as an additional feature
    if args.canonicalize == True:
        common_feature_names = common_feature_names.intersection(
            set(CANONICALIZATION.keys()))
    common_feature_names = list(common_feature_names)
    common_feature_names.append('wd_out_domain_err')
    common_feature_names = tuple(sorted(common_feature_names))

    print("Found the following common features across all datasets for OOD")
    for f in common_feature_names:
        print(f)

    all_headers = []
    all_rows = []

    for experiment_path in experiment_path_list:
        header, rows = experiment_per_dataset(all_results[experiment_path],
                                              common_feature_names,
                                              experiment_path,
                                              wd_or_ood='ood')
        all_rows.extend(rows)
        all_headers.append(header)

    with open(
            os.path.join(args.run_dir, 'sweeps_%s_canon_%s_ood.csv') %
        (datasets, args.canonicalize), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers[0])
        for r in all_rows:
            writer.writerow(r)

    common_feature_names = list(common_feature_names)
    common_feature_names.remove('wd_out_domain_err')
    common_feature_names = tuple(common_feature_names)
    print("Found the following common features across all datasets for WD")
    for f in common_feature_names:
        print(f)

    all_headers = []
    all_rows = []

    for experiment_path in experiment_path_list:
        header, rows = experiment_per_dataset(all_results[experiment_path],
                                              common_feature_names,
                                              experiment_path,
                                              wd_or_ood='wd')
        all_rows.extend(rows)
        all_headers.append(header)

    with open(
            os.path.join(args.run_dir, 'sweeps_%s_canon_%s_wd.csv') %
        (datasets, args.canonicalize), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers[0])
        for r in all_rows:
            writer.writerow(r)


if __name__ == "__main__":
    if args.canonicalize == True:
        raise NotImplementedError
    analyze_results(args.run_dir, args.sweep_name)
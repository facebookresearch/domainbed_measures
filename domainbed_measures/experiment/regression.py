"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import RidgeCV
import scipy
import itertools


class Regression:
    def __init__(self, features, target):
        self.regressor = RidgeCV(fit_intercept=False)
        self.feature_names = list(features.columns)

        features = np.array(features)
        if len(features.shape) == 1:
            features = np.array(features).reshape(-1, 1)

        self.regressor.fit(features, target)
        self.score = None

    def score_and_store(self, features, target):
        # can only be called once
        if self.score is not None:
            raise ValueError("Can only be called once")
        self.score = self.regressor.score(features, target)
        self.coef = self.regressor.coef_
        self.residuals = self.regressor.predict(features) - target
        self.spearmanr = scipy.stats.spearmanr(
            self.regressor.predict(features), target).correlation


class SubsetAnalysis:
    def __init__(self,
                 regressions,
                 datasets=None,
                 environments=None,
                 correlations=None):
        self.regressions = regressions
        self.correlations = correlations
        self.datasets = datasets
        self.environments = environments
        feature_names = [tuple(x.feature_names) for x in regressions]

        reg_meta = pd.DataFrame({
            "regression": regressions,
            "dataset": datasets,
            "environments": environments,
            "feature_name": feature_names,
            "correlations": correlations,
        })

        self.reg_meta = reg_meta

        if datasets != None and len(datasets) != len(regressions):
            raise ValueError

    def all_feature_name_subsets(self):
        return self.reg_meta['feature_name'].unique()

    def select(self, feature_name, dataset_name=None, environment_name=None):
        subset_reg_meta = self.reg_meta
        if dataset_name is not None:
            subset_reg_meta = subset_reg_meta[subset_reg_meta['dataset'] ==
                                              dataset_name]
        if feature_name is not None:
            subset_reg_meta = subset_reg_meta[subset_reg_meta['feature_name']
                                              == feature_name]
        if environment_name is not None:
            subset_reg_meta = subset_reg_meta[subset_reg_meta['environment'] ==
                                              environment_name]

        return SubsetAnalysis(
            subset_reg_meta['regression'].tolist(),
            datasets=subset_reg_meta['dataset'].tolist(),
            environments=subset_reg_meta['environments'].tolist(),
            correlations=subset_reg_meta['correlations'].tolist())

    def score(self):
        return sum([r.score for r in self.reg_meta['regression']]) / len(self)

    def weight_variance(self):
        return np.array([r.coef
                         for r in self.reg_meta['regression']]).var(0).max()

    def weight_sign_changes(self):
        return max(
            np.array([r.coef > 0 for r in self.reg_meta['regression']]).sum(),
            np.array([r.coef < 0
                      for r in self.reg_meta['regression']]).sum()) / (float(
                          len(self.reg_meta['regression']) *
                          len(self.reg_meta['regression'].iloc[0].coef)))

    def corr_without_fit(self, aggregation='mean'):
        if aggregation == 'mean':
            return np.mean([
                x.correlation for x in self.reg_meta['correlations']
                if x is not None
            ])
        elif aggregation == 'min':
            arr = [
                x.correlation for x in self.reg_meta['correlations']
                if x is not None
            ]
            if len(arr) == 0:
                return -1e3
            return np.min(arr)

    def corr_score_with_fit(self, aggregation='mean'):
        if aggregation == 'mean':
            return np.mean([r.spearmanr for r in self.reg_meta['regression']])
        elif aggregation == 'min':
            return np.min([r.spearmanr for r in self.reg_meta['regression']])

    def __len__(self):
        return len(self.reg_meta['regression'])


def get_feature_subsets(features, target, k=3):
    return [(features[list(n)], target)
            for n in itertools.combinations(features.columns, k)]


def get_datasets(data):
    datasets = sorted(list(data['dataset'].unique()))
    if "ColoredMNIST" in datasets:
        datasets.remove("ColoredMNIST")
    return datasets


def get_dataset_environments(data, dataset):
    return sorted(
        list(data.loc[data['dataset'] == dataset]['all_test_envs'].unique()))


def get_data_condition(data,
                       dataset=None,
                       environment=None,
                       feature_lambda=None,
                       normalize=False,
                       target_name="target_err"):
    if dataset is not None:
        data = data.loc[data['dataset'] == dataset]
    if environment is not None:
        data = data.loc[data['all_test_envs'] == environment]

    data = data.drop(["dataset", "all_test_envs", "test_env"], axis=1)

    # work with the normalized z-scores of targets
    target = data[target_name]
    if normalize == True:
        target = (target - target.mean()) / (target.std() +
                                             np.finfo(float).eps)
    data = data.drop([target_name], axis=1)

    if feature_lambda is not None:
        if isinstance(feature_lambda, list):
            drop_features = list(
                set(data.columns).difference(set(feature_lambda)))
        else:
            drop_features = [
                x for x in data.columns if feature_lambda(x) == False
            ]
        data = data.drop(drop_features, axis=1)

    # work with the normalized z-scores of features
    if normalize == True:
        features = (data - data.mean()) / (data.std() + np.finfo(float).eps)
    else:
        features = data

    return features, target


def get_mean_lambda_closeness(data, dataset=None, environment=None):
    lambda_columns = [x for x in data.columns if "lambda" in x]
    lambda_data = data[lambda_columns + ['dataset', 'all_test_envs']]
    if dataset != None:
        lambda_data = lambda_data[lambda_data['dataset'] == dataset]
    if environment != None:
        lambda_data = lambda_data[lambda_data['all_test_envs'] == environment]

    print(lambda_data.mean(0))


def dict_top_k(dic, key, top_k=10, reverse=True):
    return dict(
        sorted(dic.items(), key=lambda item: item[1][key],
               reverse=reverse)[:top_k])


def perform_subset_analysis(data,
                            feature_lambda=None,
                            subset_features=3,
                            single_test_env_only=False,
                            shared_weights_across_envs=False,
                            normalize=False,
                            canon=None,
                            fix_one_feat_to=None,
                            target_name='target_err'):
    print("Performing analysis on subsets of {} features".format(
        subset_features))
    print("--------------------------------------------\n")
    if single_test_env_only == True:
        data = data[data['all_test_envs'].apply(len) == 2]

    if canon is not None:
        for m in canon.index:
            data[m] = canon[m] * data[m]

    regressions = []
    datasets = []
    environments = []
    correlations = []

    common_regressors = {}
    if shared_weights_across_envs == True:
        print("Running regression across all envs.")
        all_features, all_targets = get_data_condition(
            data,
            None,
            None,
            feature_lambda=feature_lambda,
            target_name=target_name,
            normalize=normalize)
        for f, t in get_feature_subsets(all_features, all_targets,
                                        subset_features):
            common_regressors[tuple(f.columns)] = Regression(f, t)

    for dataset in get_datasets(data):
        for environment in get_dataset_environments(data, dataset):
            condition = dataset + environment
            features, target = get_data_condition(
                data,
                dataset,
                environment,
                feature_lambda=feature_lambda,
                target_name=target_name,
                normalize=normalize)
            for f, t in get_feature_subsets(features, target, subset_features):
                if fix_one_feat_to is not None:
                    if fix_one_feat_to not in tuple(f.columns):
                        continue
                if shared_weights_across_envs == True:
                    this_common_regressor = common_regressors[tuple(f.columns)]
                    regressor = copy.copy(this_common_regressor)
                else:
                    regressor = Regression(f, t)
                regressor.score_and_store(f, t)

                if subset_features == 1:
                    correlations.append(scipy.stats.spearmanr(f, t))
                else:
                    correlations.append(None)
                regressions.append(regressor)
                datasets.append(dataset)
                environments.append(environment)

    analysis = SubsetAnalysis(regressions, datasets, environments,
                              correlations)

    return analysis


def report_top_subsets(analysis,
                       filter_dataset,
                       sort_by,
                       top_k=100,
                       canonicalize=False):
    results = {}

    for subset in analysis.all_feature_name_subsets():
        analysis_subset = analysis.select(feature_name=subset,
                                          dataset_name=filter_dataset)
        results[subset] = {
            'score': analysis_subset.score(),
            'weight_variance': analysis_subset.weight_variance(),
            'times_same_sign': analysis_subset.weight_sign_changes(),
            'correlation_no_fit': analysis_subset.corr_without_fit(),
            'correlation_with_fit': analysis_subset.corr_score_with_fit(),
        }

    print("Best subsets on %s according to %s score:" %
          (filter_dataset, sort_by))

    results_df = pd.DataFrame(results)
    for k, v in dict_top_k(results, sort_by, top_k=top_k,
                           reverse=True).items():
        print(
            "{0}] R2:{1:.3f}| corr: {2:.3f} | corr/w/fit {3:.3f} times_same_sign: {4}"
            .format(
                k,
                v['score'],
                v['correlation_no_fit'],
                v['correlation_with_fit'],
                v['times_same_sign'],
            ))

    results = results_df.transpose()
    results['measure'] = [str(x[0]) for x in results.index]
    results = results.reset_index()

    canon = None
    if canonicalize == True:
        canon = (results['correlation_no_fit'] > 0).astype(float)
        canon[canon == 0.0] = -1.0
        results['correlation_no_fit'] = canon * results['correlation_no_fit']
        results[
            'correlation_with_fit'] = canon * results['correlation_with_fit']
        canon.index = results['measure']

    return results, canon
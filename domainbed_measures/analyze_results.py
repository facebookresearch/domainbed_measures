"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Some tools for analyzing results and plotting."""
import argparse
import pandas as pd

from domainbed_measures.experiment.regression import perform_subset_analysis
from domainbed_measures.experiment.regression import report_top_subsets

parser = argparse.ArgumentParser()

parser.add_argument("--input_csv",
                    default="",
                    help="Directory with the runs and results.")

parser.add_argument("--dataset",
                    default="RotatedMNIST",
                    help="Dataset for which we are running the analysis")

parser.add_argument("--sort_by",
                    default="correlation_with_fit",
                    help="whether to sort by the rank correlation after "
                    "fitting the regression coefficients or by "
                    "weighting all features equally")

parser.add_argument(
    "--stratified_or_joint",
    default="joint",
    help="Whether to have one set of weights across"
    "all environments (joint setting) or different weights"
    "for each set of training environments (stratified setting)")

parser.add_argument("--top_k",
                    default=20,
                    type=int,
                    help="Number of top feature subsets to print")

parser.add_argument(
    "--num_features",
    default=1,
    type=int,
    help="Number of feature combinations to use for regression.")

parser.add_argument("--fix_one_feature_to_wd",
                    default="False",
                    action="store_true",
                    help="Set one of the features for regression to be"
                    " the validation accuracy in domain")

args = parser.parse_args()

if __name__ == "__main__":
    data = pd.read_csv(args.input_csv)
    analysis = perform_subset_analysis(
        data,
        subset_features=args.num_features,
        normalize=True,
        fix_one_feat_to="wd_out_domain_err"
        if args.fix_one_feature_to_wd == True else None,
        shared_weights_across_envs=args.stratified_or_joint == "joint")

    report_top_subsets(analysis,
                       filter_dataset=args.dataset,
                       sort_by=args.sort_by,
                       top_k=args.top_k)
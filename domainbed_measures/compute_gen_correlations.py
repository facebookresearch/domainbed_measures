"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Load a set of models, and compute correlation with generalization measure.
"""
import argparse
import pandas as pd
import os
import submitit
import string
import logging
import itertools
import time
import random
import json

from collections import defaultdict

from domainbed_measures.measures.registry import MeasureRegistry
from domainbed_measures.experiment.experiment import Experiment
from domainbed_measures.experiment.experiment import VarianceExperiment

MAX_JOBS_IN_ARRAY = 128
_SLEEP_TIME = 1200
_RANDOM_SEED = 44

_DATASET_TO_MEMORY = defaultdict(lambda: "95GB")
_DATASET_TO_MEMORY["RotatedMNIST"] = "64GB"
_DATASET_TO_MEMORY["ColoredMNIST"] = "64GB"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

MEASURES = MeasureRegistry._VALID_MEASURES

parser = argparse.ArgumentParser()

parser.add_argument("--debug",
                    action="store_true",
                    help="Whether to run in debug mode")
parser.add_argument("--debug_model_path",
                    default="",
                    type=str,
                    help="Run debug job on a particular model run")
parser.add_argument("--algorithm", default="ERM", help="Algorithm to analyze")
parser.add_argument("--dataset",
                    default="RotatedMNIST",
                    help="Dataset to analyze")
parser.add_argument("--job_done_file",
                    default="",
                    help="Use write_job_status_file.py to generate.")
parser.add_argument("--run_dir", default="")
parser.add_argument(
    "--measures",
    default="domainbed_measures/scratch/measure_list_neurips.json",
    help=
    "Comma separated list of measures to compute, all measures computed if empty"
)
parser.add_argument("--device",
                    default="cuda",
                    help="Device on which to run experiments")
parser.add_argument("--slurm_timeout_min",
                    default=600,
                    type=int,
                    help="SLURM job timeout in min")
parser.add_argument("--slurm_partition",
                    default="learnlab",
                    help="Parition for slurm jobs.")
parser.add_argument("--job_str",
                    default="",
                    type=str,
                    help="Optional job string")
parser.add_argument(
    "--dirty_ood_split",
    type=str,
    default="out",
    help="Can be either out or in,"
    "if out then the generalization measure is computed on the out split otherwise on the"
    "in split. Also, if generalization is computed on out split, then generalization"
    "performance is computed on the in split and vice versa.")
parser.add_argument(
    "--all_measures_one_job",
    action='store_true',
    default=False,
    help="Whether to launch a new job for each measure or compute"
    "all measures for a run in one job.")
parser.add_argument("--calc_variance",
                    action="store_true",
                    help="Whether to calculate measures or variance.")
parser.add_argument("--max_num_jobs",
                    default=-1,
                    type=int,
                    help="Max jobs to use.")

args = parser.parse_args()


def main(args):

    if args.run_dir == "":
        raise ValueError(
            "Please provide a working directory for storing generalization measure values."
        )

    MODEL_FILTERS = {
        "algorithm": args.algorithm,
        "dataset": args.dataset,
        "status": "done"
    }

    models_and_info = pd.read_csv(
        args.job_done_file,
        delimiter=" ",
        names=["path", "algorithm", "dataset", "status"])

    for filter, value in MODEL_FILTERS.items():
        models_and_info = models_and_info[models_and_info[filter] == value]

    if args.job_str == '' and args.debug == True:
        job_str = 'debug'
    else:
        job_str = args.job_str

    out_folder = os.path.join(
        args.run_dir,
        args.job_done_file.rstrip('.txt').split("/")[-1], "%s_%s_%s_%s" %
        (job_str, args.dirty_ood_split, args.algorithm, args.dataset))

    logging.info(f"Using directory {out_folder} for storing runs")

    if args.device == "cuda":
        gpus_per_node = 1
    else:
        gpus_per_node = 0

    if args.measures == "":
        measures_to_compute = MeasureRegistry._VALID_MEASURES
    elif '.json' in args.measures:
        logging.info('Using measure list file %s' % (args.measures))
        with open(args.measures, 'r') as f:
            measures_to_compute = json.load(f)
    else:
        measures_to_compute = args.measures.split(",")

    if args.calc_variance == True:
        experiment_to_use = VarianceExperiment(
            dirty_ood_split=args.dirty_ood_split, )
    else:
        experiment_to_use = Experiment(dirty_ood_split=args.dirty_ood_split, )

    jobs = []

    model_paths = list(models_and_info["path"])
    if args.debug_model_path != "":
        model_paths = [args.debug_model_path]

    if args.all_measures_one_job:
        all_jobs = list(model_paths)
    else:
        all_jobs = list(itertools.product(model_paths, measures_to_compute))

    current_idx = 0
    current_jobs_in_array = 0

    # Set random seed for file directory names
    random.seed(_RANDOM_SEED)
    # Ensure we never place more jobs in a job array than can be run concurrently
    while (current_idx < len(all_jobs)):

        if args.max_num_jobs != -1 and current_idx >= args.max_num_jobs:
            break

        job_path = os.path.join(
            out_folder, 'slurm_files', ''.join(
                random.choices(string.ascii_lowercase + string.digits, k=10)))
        logging.info(f"Launching jobs with path {job_path}")

        ex = submitit.AutoExecutor(
            job_path,
        )

        if args.slurm_partition != "":
            ex.update_parameters(
                slurm_partition=args.slurm_partition,
                gpus_per_node=gpus_per_node,
                cpus_per_task=4,
                nodes=1,
                timeout_min=args.slurm_timeout_min,
                slurm_mem=_DATASET_TO_MEMORY[args.dataset],
            )

        with ex.batch():
            for idx in range(current_idx, len(all_jobs)):
                if args.max_num_jobs != -1 and idx >= args.max_num_jobs:
                    break
                if args.all_measures_one_job == True:
                    path = all_jobs[idx]
                    measure = measures_to_compute
                else:
                    path, measure = all_jobs[idx]

                if args.debug or args.slurm_partition == "":
                    experiment_to_use(path, measure, args.dataset, job_path)
                    if args.debug:
                        break
                else:
                    jobs.append(
                        ex.submit(experiment_to_use, path, measure,
                                  args.dataset, job_path))
                    current_jobs_in_array += 1

                if current_jobs_in_array >= MAX_JOBS_IN_ARRAY:
                    logging.info(f"Starting new job array..at {idx+1}")
                    current_idx = idx + 1
                    current_jobs_in_array = 0
                    break

                if len(jobs) > len(all_jobs):
                    raise ValueError

            current_idx = idx + 1
        if args.debug:
            break

    logging.info("Launching %d jobs for %s" % (len(jobs), args.dataset))

    start_time = time.time()
    while not all([j.done() for j in jobs]):
        time.sleep(_SLEEP_TIME)
        jobs_done = sum([j.done() for j in jobs])
        logging.info("%d/%d jobs done (%f sec per job)" %
                     (jobs_done, len(jobs),
                      (time.time() - start_time) / (jobs_done + 1)))

    _ = [j.result() for j in jobs]


if __name__ == "__main__":
    main(args)

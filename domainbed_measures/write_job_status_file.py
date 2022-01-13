"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Write a status file for sweeps.

Outputs a file with all the jobs in a given run, along with details of
the algorithm the job uses, whethter it is done or not and the dataset
the particular run is on, for the DomainBed jobs.
"""
import argparse
import os
import jsonlines

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--sweep_dir",
                    type=str,
                    default="/checkpoint/dlp/sweep_iclr",
                    help="Directory with all the sweeps")
parser.add_argument("--output_txt",
                    type=str,
                    default="scratch/jobs_sweep_iclr.txt",
                    help="Output txt with sweep status")

args = parser.parse_args()


def get_status_file(sweep_dir, output_txt):
    out_lines = []
    num_done = 0
    total = 0
    failed = 0

    for jobname in tqdm(os.listdir(sweep_dir)):
        if not os.path.isdir(os.path.join(sweep_dir, jobname)):
            continue

        out_txt = os.path.join(sweep_dir, jobname, 'out.txt')

        if not os.path.exists(out_txt):
            print(out_txt + "does not exist")
            continue

        with open(out_txt, 'r') as f:
            at = f.readlines()
            dataset = at[12].split(":")[-1].lstrip().rstrip()

        if os.path.exists(os.path.join(sweep_dir, jobname, 'results.jsonl')):
            total += 1
            with jsonlines.open(
                    os.path.join(sweep_dir, jobname, "results.jsonl"),
                    'r') as reader:
                for obj in reader:
                    algorithm = obj['args']['algorithm']
                    dataset = obj['args']['dataset']
                    break
            if os.path.exists(os.path.join(sweep_dir, jobname, 'done')):
                with open(os.path.join(sweep_dir, jobname, 'done'), 'r') as f:
                    done = f.readline()

                num_done += int(done == 'done')
            out_lines.append(
                "%s %s %s %s\n" %
                (os.path.join(sweep_dir, jobname), algorithm, dataset, done))
        else:
            failed += 1

    print("{%d/%d} jobs are done, {%d} failed" % (num_done, total, failed))

    with open(output_txt, 'w') as f:
        f.writelines(out_lines)


if __name__ == "__main__":
    get_status_file(args.sweep_dir, args.output_txt)

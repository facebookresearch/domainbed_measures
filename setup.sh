#Copyright (c) Meta Platforms, Inc. and affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

# Setup some useful filepaths

# Please run this script after running `source init.sh`
# Path where the jobs and sweeps run using the DomainBed code are to be stored
export DOMAINBED_RUN_DIR=""

# Path where you put the datasets corresponding to DomainBed when training the models
export DOMAINBED_DATA=""

# Path where you want to store the measures that this codebase computes
export MEASURE_RUN_DIR=""

# Set the partition you want to use for SLURM to run it on a cluster
# If empty, the measure computation will run locally
export SLURM_PARTITION=""

# Path where the DomainBed codebase exists, by default it is in /path/to/project/DomainBed
# after running init.sh
export DOMAINBED_PATH="${PWD}/DomainBed"

export PYTHONPATH="${PYTHONPATH}:${DOMAINBED_PATH}"
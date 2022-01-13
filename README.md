# Code for "An Empirical Investigation of Domian Generalization with Empirical Risk Minimizers" (NeurIPS 2021)

## Motivation and Introduction
Domain Generalization is a task in machine learning where
given a shift in the input data distribution, one is expected to perform 
well on a test task with a different input data distribution. For example, 
one might train a digit classifier on MNIST data and ask the model
to generalize to predict digits that are rotated by say 30 degrees.

While many approaches have been proposed for this problem, we were intrigued by the results on the [DomainBed](github.com/facebookresearch/domainbed) benchmark which suggested that using the
simple, empirical risk minimization (ERM) with a proper hyperparameter
sweep leads to performance close to state of the art on Domain
Generalization Problems.

What governs the generalization of a trained deep learning model
using ERM to a given data distribution? This is the question we seek
to answer in our NeurIPS 2021 paper:

**An Empirical Investigation of Domain Generalization with Empirical Risk Minimizers.** Rama Vedantam, David Lopez-Paz\*, David Schwab\*.

NeurIPS 2021 (\*=Equal Contribution)

This repository contains code used for producing the results in our
paper.
## Initial Setup
1. Run `source init.sh` to install all the dependencies for the project. This will also initialize DomainBed
as a submodule for the project 

2. Set requisite paths in `setup.sh`, and run `source setup.sh`
## Computing Generalization Measures
* Get set up with the [DomainBed](https://github.com/facebookresearch/domainbed) codebase and launch a sweep
for an initial set of trained models (illustrated below
for rotated MNIST dataset):

```
cd DomainBed/

python -m domainbed.scripts.sweep launch\
       --data_dir=${DOMAINBED_DATA} \
       --output_dir=${DOMAINBED_RUN_DIR}/sweep_fifty_fifty \
       --algorithms=ERM \
       --holdout_fraction=0.5\
       --datasets=RotatedMNIST \
       --n_hparams=1\
       --command_launcher submitit
```
After this step, we have a set of trained models that we can now
look to evaluate and measure. Note that unlike the original [domainbed paper](https://arxiv.org/abs/2007.01434)
we holdout a larger fraction (50%) of the data for evaluation of the measures.

* Once the sweep finishes, aggregate the different files for use by the `domianbed_measures` codebase:
```
python domainbed_measures/write_job_status_file.py \
                --sweep_dir=${DOMAINBED_RUN_DIR}/sweep_fifty_fifty \
                --output_txt="domainbed_measures/scratch/sweep_release.txt"
```

* Once this step is complete, we can compute various generalization measures and store them to disk for future analysis
using:
```
SLURM_PARTITION="TO_BE_SET"
python domainbed_measures/compute_gen_correlations.py \
	--algorithm=ERM \
    --job_done_file="domainbed_measures/scratch/sweep_release.txt" \
    --run_dir=${MEASURE_RUN_DIR} \
    --all_measures_one_job \
	--slurm_partition=${SLURM_PARTITION}
```
Where we utilize [slurm](https://slurm.schedmd.com/documentation.html) on a compute cluster to scale the experiments to thousands of models. If you do not have access to such a cluster with multiple GPUs to parallelize the computation, use `--slurm_partition=""` above and the code will run on a single GPU (although the results might take a long time to compute!).

* Finally, once the above code is done, use the following code snippet to aggregate the values of the different generalization measures:
```
python domainbed_measures/extract_generalization_features.py \
    --run_dir=${MEASURE_RUN_DIR} \
    --sweep_name="_out_ERM_RotatedMNIST"
```

This step yeilds `.csv` files where each row corresponds to a given
trained model. Each row overall has the following format:
```
dataset | test_envs | measure 1 | measure 2 | measure 3 | target_err
```
where:
* `test_envs` specifies which environments the model is tested on or
equivalently trained on, since the remaining environments are used for
training
* `target_err` specifies the target error value for regression
* `measure 1` specifies the which measure is being computed, e.g.
sharpness or fisher eigen value based measures

In case of the file named, for example, `sweeps__out_ERM_RotatedMNIST_canon_False_ood.csv`, the validation
error within domain `wd_out_domain_err` is also used as one of
the `measures` and `target_err` is the out of domain generalization
error, and all measures are computed on a held-out set of image
inputs from the target domain
(for more details see the [paper](https://openreview.net/forum?id=Z8mLxlpSyrJ&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions))).

Alternatively, in case of the file named, 
`sweeps__out_ERM_RotatedMNIST_canon_False_wd.csv`, the `target_err`
is the validation accuracy in domain, and all the measures are computed
on the in-distribution held-out images.

* Using this file one can do a number of interesting
regression analyses as reported in the paper for measuring
generalization.

For example, to generate the kind of results
in Table. 1 of the [paper](https://openreview.net/forum?id=Z8mLxlpSyrJ&referrer=%5BAuthor%20Console%5D\(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions) in the *joint*
setting, run the following command options:
```
python domainbed_measures/analyze_results.py \
    --input_csv="${MEASURE_RUN_DIR}/sweeps__out_ERM_RotatedMNIST_canon_False_ood.csv"\
    --stratified_or_joint="joint"\
    --num_features=2 \
    --fix_one_feature_to_wd
```

Alternatively, to generate results in the *stratified* setting, run:
```
python domainbed_measures/analyze_results.py \
    --input_csv="${MEASURE_RUN_DIR}/sweeps__out_ERM_RotatedMNIST_canon_False_ood.csv"\
    --stratified_or_joint="stratified"\
    --num_features=2 \
    --fix_one_feature_to_wd
```

Finally, to generate results using a single feature (Alone setting in Table. 1), run:
```
python domainbed_measures/analyze_results.py \
    --input_csv="${MEASURE_RUN_DIR}/sweeps__out_ERM_RotatedMNIST_canon_False_ood.csv"\
    --num_features=1
```

## Translation of measures from the code to the paper
The following table illustrates all the measures in the [paper](https://openreview.net/attachment?id=Z8mLxlpSyrJ&name=supplementary_material) (Appendix Table. 2) and how they are referred
to in the codebase:

| Measure Name | Code Reference |
|----------|----------------|
|H-divergence | `c2st`|
|H-divergence + Source Error | `c2st_perr`|
|H-divergence MS | `c2st_per_env` |
|H-divergence MS + Source Error | `c2st_per_env_perr`|
|H-divergence (train) | `c2st_train` |
|H-divergence (train) + Source Error | `c2st_train_perr` |
|H-divergence (train) MS | `c2st_train_per_env`|
|Entropy-Source or Entropy | `entropy`|
|Entropy-Target | `entropy_held_out` |
|Fisher-Eigval-Diff | `fisher_eigval_sum_diff_ex_75`|
|Fisher-Eigval | `fisher_eigval_sum_ex_75` |
|Fisher-Align or Fisher (main paper) | `fisher_eigvec_align_ex_75`|
|HΔH-divergence SS | `hdh`|
|HΔH-divergence SS + Source Error | `hdh_perr`|
|HΔH-divergence MS | `hdh_per_env`|
|HΔH-divergence MS + Source Error | `hdh_per_env_perr`|
|HΔH-divergence (train) SS | `hdh_train`|
|HΔH-divergence (train) SS + Source Error| `hdh_train_perr`|
|Jacobian| `jacobian_norm`|
|Jacobian Ratio| `jacobian_norm_relative`|
|Jacobian Diff| `jacobian_norm_relative_diff`|
|Jacobian Log Ratio | `jacobian_norm_relative_log_diff`|
|Mixup| `mixup`|
|Mixup Ratio | `mixup_relative`|
|Mixup Diff| `mixup_relative_diff`|
|Mixup Log Ratio| `mixup_relative_log_diff`|
|MMD-Gaussian| `mmd_gaussian`|
|MMD-Mean-Cov| `mmd_mean_cov`|
|L2-Path-Norm.| `path_norm`|
|Sharpness| `sharp_mag`|
|H+-divergence SS| `v_plus_c2st`|
|H+-divergence SS + Source Error | `v_plus_c2st_perr`|
|H+-divergence MS| `v_plus_c2st_per_env`|
|H+-divergence MS + Source Error| `v_plus_c2st_per_env_perr`|
|H+ΔH+-divergence SS| `v_plus_hdh`|
|H+ΔH+-divergence SS + Source Error| `v_plus_hdh_perr`|
|H+ΔH+-divergence MS| `v_plus_hdh_per_env`|
|H+ΔH+-divergence MS + Source Error| `v_plus_hdh_per_env_perr`|
|Source Error| `wd_out_domain_err`| 


## Acknowledgments
We thank the developers of [Decodable Information Bottleneck](https://github.com/facebookresearch/decodable_information_bottleneck),
[Domain Bed](https://github.com/facebookresearch/domainbed) and [Jonathan Frankle](https://www.csail.mit.edu/person/jonathan-frankle)
for code we found useful for this project.

## License
This source code is released under the Creative Commons Attribution-NonCommercial 
4.0 International license, included [here](https://github.com/facebookresearch/domainbed_measures/blob/main/LICENSE.md).
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import defaultdict

from domainbed_measures.measures.classical import EntropyOutput
from domainbed_measures.measures.classical import PathNorm
from domainbed_measures.measures.classical import SharpnessMagnitude
from domainbed_measures.measures.held_out_measures import FunctionVMinimality
from domainbed_measures.measures.held_out_measures import ClassifierTwoSampleTest
from domainbed_measures.measures.held_out_measures import JacobianNorm
from domainbed_measures.measures.held_out_measures import ValidationAccuracy
from domainbed_measures.measures.held_out_measures import Mixup
from domainbed_measures.measures.held_out_measures import MixupRelative
from domainbed_measures.measures.held_out_measures import MixupRelativeDiff
from domainbed_measures.measures.held_out_measures import MixupRelativeLogDiff
from domainbed_measures.measures.held_out_measures import JacobianNormRelative
from domainbed_measures.measures.held_out_measures import JacobianNormRelativeLogDiff
from domainbed_measures.measures.held_out_measures import JacobianNormRelativeDiff
from domainbed_measures.measures.held_out_measures import HDelHDivergence
from domainbed_measures.measures.held_out_measures import MMD
from domainbed_measures.measures.fisher import FisherEigValues
from domainbed_measures.measures.fisher import FisherEigValuesSumDiff
from domainbed_measures.measures.fisher import FisherEigVecAlign

_MNIST_NUM_EXAMPLES_FISHER = 200


class MeasureRegistry(object):
    _KWARGS = defaultdict(dict)
    # "data_args" here configure the dataloader the way we want it
    _KWARGS["entropy"] = {
        "measure_args": {
            "convert_bn_to_conv": True,
            'train_union_loader_type': 'longest'
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["entropy_held_out"] = {
        "measure_args": {
            "convert_bn_to_conv": True,
            'train_union_loader_type': 'longest',
            'use_eval_data': True,
            'compute_on': 'held_out',
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["path_norm"] = {
        "measure_args": {
            "convert_bn_to_conv": True
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["sharp_mag"] = {
        "measure_args": {
            "convert_bn_to_conv": True,
            "needs_training": True
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["v_minimality"] = {
        "measure_args": {
            "needs_training": True,
            "train_epochs": 20,
            "cond_min": False,
            "recompute_features_every_epoch": False,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["v_plus_minimality"] = {
        "measure_args": {
            "needs_training": True,
            "cond_min": False,
            "train_epochs": 20,
            "v_plus": True,
            "recompute_features_every_epoch": False,
        },
        "data_args": {
            "include_index": True
        }
    }
    _KWARGS["cond_v_minimality"] = {
        "measure_args": {
            "needs_training": True,
            "train_epochs": 30,
            "cond_min": True,
            "recompute_features_every_epoch": False,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["cond_v_plus_minimality"] = {
        "measure_args": {
            "needs_training": True,
            "train_epochs": 30,
            "cond_min": True,
            "v_plus": True,
            "recompute_features_every_epoch": False,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["jacobian_norm"] = {
        "measure_args": {
            "needs_training": True,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["jacobian_norm_relative"] = {
        "measure_args": {
            "needs_training": True,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["jacobian_norm_relative_diff"] = {
        "measure_args": {
            "needs_training": True,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["jacobian_norm_relative_log_diff"] = {
        "measure_args": {
            "needs_training": True,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["validation_accuracy"] = {
        "measure_args": {
            "needs_training": False,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["mixup"] = {
        "measure_args": {
            "needs_training": False,
            "alpha": 0.1,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["mixup_alpha_0.3"] = {
        "measure_args": {
            "needs_training": False,
            "alpha": 0.3,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["mixup_relative"] = {
        "measure_args": {
            "needs_training": False,
            "alpha": 0.1,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["mixup_relative_diff"] = {
        "measure_args": {
            "needs_training": False,
            "alpha": 0.1,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["mixup_relative_log_diff"] = {
        "measure_args": {
            "needs_training": False,
            "alpha": 0.1,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["mixup_alpha_0.3_relative"] = {
        "measure_args": {
            "needs_training": False,
            "alpha": 0.3,
            "use_eval_data": True,
            'train_union_loader_type': 'longest',
            "convert_bn_to_conv": False
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["hdh"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "train_or_test": 'test',
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["v_plus_hdh"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "train_or_test": 'test',
            "v_plus": True,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["v_plus_hdh_per_env"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "train_or_test": 'test',
            "v_plus": True,
            "per_env": True,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["hdh_per_env"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "train_or_test": 'test',
            "per_env": True,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["hdh_train"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "train_or_test": 'train',
        },
        "data_args": {
            "include_index": True
        }
    }

    # Include index is true for train test classifier because it inherits from vminimality
    # As such it does not need the index per se
    _KWARGS["c2st"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "train_or_test": 'test',
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["c2st_train"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            'train_or_test': 'train',
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["c2st_train_per_env"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            'train_or_test': 'train',
            'per_env': True,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["c2st_per_env"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            'train_or_test': 'test',
            'per_env': True,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["v_plus_c2st_per_env"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            'train_or_test': 'test',
            'per_env': True,
            "v_plus": True,
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["v_plus_c2st"] = {
        "measure_args": {
            "needs_training": True,
            'train_union_loader_type': 'longest',
            'use_eval_data': True,
            "train_epochs": 50,
            "v_plus": True,
            'train_or_test': 'test',
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["mmd_gaussian"] = {
        "measure_args": {
            "needs_training": False,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "kernel_type": "gaussian"
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["mmd_mean_cov"] = {
        "measure_args": {
            "needs_training": False,
            'train_union_loader_type': 'longest',
            "use_eval_data": True,
            "train_epochs": 50,
            "kernel_type": "mean_cov"
        },
        "data_args": {
            "include_index": True
        }
    }

    _KWARGS["fisher_eigval_sum_ex_75"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": -1,
            "max_num_examples": 75,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["fisher_eigval_sum_ex_40"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": -1,
            "max_num_examples": 40,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["fisher_eigval_sum_diff_ex_75"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": -1,
            "max_num_examples": 75,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["fisher_eigval_sum_diff_ex_40"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": -1,
            "max_num_examples": 40,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["fisher_eigval_sum_ex_3000"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": 1000,
            "max_num_examples": 3000,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["fisher_eigvec_align_ex_75"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": -1,
            "max_num_examples": 75,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["fisher_eigvec_align_ex_40"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": -1,
            "max_num_examples": 40,
        },
        "data_args": {
            "include_index": False
        }
    }

    _KWARGS["fisher_eigvec_align_ex_3000"] = {
        "measure_args": {
            "needs_training": True,
            "train_union_loader_type": 'longest_padded',
            "use_eval_data": True,
            "num_eig": 1000,
            "max_num_examples": 3000,
        },
        "data_args": {
            "include_index": False
        }
    }

    _VALID_MEASURES = list(_KWARGS.keys())

    def __getitem__(self, measure="entropy"):
        if measure not in self._VALID_MEASURES:
            raise NotImplementedError

        if measure == "entropy" or measure == 'entropy_held_out':
            return EntropyOutput
        elif measure == "path_norm":
            return PathNorm
        elif measure == "sharp_mag":
            return SharpnessMagnitude
        elif (measure == "v_minimality" or measure == "v_plus_minimality"
              or measure == "cond_v_minimality"
              or measure == "cond_v_plus_minimality"):
            return FunctionVMinimality
        elif measure == "jacobian_norm":
            return JacobianNorm
        elif measure == "jacobian_norm_relative":
            return JacobianNormRelative
        elif measure == "jacobian_norm_relative_log_diff":
            return JacobianNormRelativeLogDiff
        elif measure == "jacobian_norm_relative_diff":
            return JacobianNormRelativeDiff
        elif measure == "validation_accuracy":
            return ValidationAccuracy
        elif measure == "mixup" or measure == "mixup_alpha_0.3":
            return Mixup
        elif measure == "mixup_relative" or measure == "mixup_alpha_0.3_relative":
            return MixupRelative
        elif measure == "mixup_relative_diff":
            return MixupRelativeDiff
        elif measure == "mixup_relative_log_diff":
            return MixupRelativeLogDiff
        elif "fisher_eigval_sum_diff" in measure:
            return FisherEigValuesSumDiff
        elif "fisher_eigval_sum" in measure:
            return FisherEigValues
        elif "fisher_eigvec_align" in measure:
            return FisherEigVecAlign
        elif "c2st" in measure:
            return ClassifierTwoSampleTest
        elif "mmd" in measure:
            return MMD
        elif 'hdh' in measure:
            return HDelHDivergence

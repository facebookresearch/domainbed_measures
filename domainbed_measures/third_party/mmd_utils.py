"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

MMD code from github.com/facebookresearch/DomainBed
"""
import torch


def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1),
                      x1,
                      x2.transpose(-2, -1),
                      alpha=-2).add_(x1_norm)
    return res.clamp_min_(1e-30)


def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    D = my_cdist(x, y)
    K = torch.zeros_like(D)

    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))

    return K


def mmd(x, y, kernel_type="gaussian"):
    if kernel_type == "gaussian":
        Kxx = gaussian_kernel(x, x).mean()
        Kyy = gaussian_kernel(y, y).mean()
        Kxy = gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy
    elif kernel_type == "mean_cov":
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff
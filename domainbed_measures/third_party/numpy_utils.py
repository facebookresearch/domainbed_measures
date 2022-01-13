"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Some utilities for numpy.

Author: Jonathan Frankle
"""
import numpy as np


# Randomize a numpy array.
def randomize_array(arr):
    shape = arr.shape
    arr = arr.flatten()
    perm = np.random.permutation(arr.size)
    arr = arr[perm]
    return arr.reshape(shape)


def vectorize(weight_dict, key_whitelist=None):
    """Cannonically converts a dictionary (layer name, numpy value) to a single vector."""
    return np.concatenate([
        weight_dict[k].flatten() for k in sorted(weight_dict.keys())
        if key_whitelist is None or k in key_whitelist
    ])


def unvectorize(dict_like, vec):
    output = {}
    for name in sorted(dict_like.keys()):
        if vec.size == 0:
            raise ValueError('Ran out of parameters for unvectorizing.')

        size, shape = dict_like[name].size, dict_like[name].shape
        this_vec = vec[:size]
        vec = vec[size:]

        output[name] = this_vec.reshape(shape)

    if vec.size != 0:
        raise ValueError('Not all parameters unvectorized properly.')

    return output


def magnitude_prune_vector(vec, pct, mask=None, keep_lowest=False):
    if mask is None: mask = np.ones_like(vec)
    coeff = -1 if keep_lowest else 1

    vec = np.nan_to_num(vec)
    threshold_index = np.ceil(vec[mask == 1].size * pct).astype(int)
    threshold = np.sort(coeff * np.abs(vec[mask == 1]))[threshold_index]
    mask = np.where(coeff * np.abs(vec) > threshold, mask, np.zeros_like(vec))
    return mask


# Mask dict is the mask dictionary whose layerwise percentages the new mask should match.
def magnitude_prune_layerwise(weights_dict,
                              pcts_dict,
                              masks_dict=None,
                              keep_lowest=False):
    if masks_dict is None:
        masks_dict = {k: np.ones_like(v) for k, v in weights_dict.items()}

    out = {}
    for k in masks_dict:
        shape = masks_dict[k].shape
        out[k] = magnitude_prune_vector(weights_dict[k].reshape(-1),
                                        pcts_dict[k],
                                        masks_dict[k].reshape(-1), keep_lowest)
        out[k] = out[k].reshape(shape)

    return out


def angle(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    if vec1.dot(vec2) > .99:
        return 0
    return np.rad2deg(np.arccos(vec1.dot(vec2)))


def reinflate(masked_vec, mask_vec, zero_vec=None):
    if zero_vec is None: zero_vec = np.zeros(mask_vec.size - masked_vec.size)
    out = []
    index_vec = 0
    index_zero = 0
    for index_mask in range(len(mask_vec)):
        if mask_vec[index_mask]:
            out.append(masked_vec[index_vec])
            index_vec += 1
        else:
            out.append(zero_vec[index_zero])
            index_zero += 1
    return np.array(out)


def project(orthonormal_basis, other):
    if len(other.shape) == 1: other = np.expand_dims(other, -1)

    scalar_projections = other.T.dot(orthonormal_basis)
    return orthonormal_basis.dot(scalar_projections.T)


# Use the "Tiny Subspace" measure of similarity of a vector or basis to this basis.
def subspace_similarity(orthonormal_basis, other):
    other_norm = np.linalg.norm(other, axis=0)
    other = np.divide(other,
                      other_norm,
                      out=np.zeros_like(other),
                      where=other_norm != 0)
    vector_projections = project(orthonormal_basis, other)

    original_magnitudes = np.linalg.norm(other, axis=0)
    projected_magnitudes = np.linalg.norm(vector_projections, axis=0)
    ratio = projected_magnitudes[
        original_magnitudes != 0] / original_magnitudes[
            original_magnitudes != 0]

    if ratio.size > 0: return np.mean(ratio)
    else: return 0


# Welford's method for streaming computation of standard deviation.
class RunningStats:
    def __init__(self):
        self.M = None
        self.S = None
        self.count = 0
        self.running_mean = 0

    def add(self, v):
        if isinstance(v, list): v = np.array(v)
        v = v.astype(float)

        self.count += 1
        if self.count == 1:
            self.M = v
            self.S = np.zeros_like(v)
            self.running_mean = v
        else:
            prev_M = self.M.copy()
            self.M += (v - prev_M) / self.count
            self.S += (v - prev_M) * (v - self.M)
            self.running_mean = self.running_mean * (
                self.count - 1) / self.count + v / self.count

    @property
    def mean(self):
        return self.running_mean

    @property
    def std(self):
        return np.sqrt(self.S / (self.count))
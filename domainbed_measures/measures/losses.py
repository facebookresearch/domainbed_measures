"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NegHDelHCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super(NegHDelHCriterion, self).__init__()
        if reduction != 'mean':
            raise ValueError("Reduction can only be mean")
        self.reduction = reduction

    def forward(self, logit_list, domain_labels):
        if torch.max(domain_labels) > 1:
            raise ValueError(
                "Only expect binary domain labels, encoded starting at 0")

        labels_for_network_1 = torch.argmax(logit_list[1], dim=-1)
        labels_for_network_2 = torch.argmax(logit_list[0], dim=-1)

        cat_pr_1 = torch.ones_like(logit_list[0])
        cat_pr_1[torch.arange(0, labels_for_network_1.shape[-1]),
                 labels_for_network_1] = 0
        random_labels_1 = torch.multinomial(cat_pr_1, 1).squeeze()

        cat_pr_2 = torch.ones_like(logit_list[0])
        cat_pr_2[torch.arange(0, labels_for_network_2.shape[-1]),
                 labels_for_network_2] = 0
        random_labels_2 = torch.multinomial(cat_pr_2, 1).squeeze()

        final_labels_for_network_1 = torch.where(domain_labels.bool(),
                                                 random_labels_1,
                                                 labels_for_network_1)

        final_labels_for_network_2 = torch.where(domain_labels.bool(),
                                                 random_labels_2,
                                                 labels_for_network_2)

        log_probs_net_1 = F.log_softmax(logit_list[0], dim=-1)
        log_probs_net_2 = F.log_softmax(logit_list[1], dim=-1)
        batch_shape = logit_list[0].shape[0]

        # Increase disagreement on source domain, by minimizing the logprobs
        # of correct label, and maximizing the logprobs of incorrect label
        loss = -1 * torch.mean(
            log_probs_net_1[torch.arange(0, batch_shape).long(),
                            final_labels_for_network_1] +
            log_probs_net_2[torch.arange(0, batch_shape).long(),
                            final_labels_for_network_2])
        return loss

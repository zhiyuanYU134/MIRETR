import math

import torch
import torch.nn as nn

from .functional import log_sinkhorn_normalization


class LearnableLogOptimalTransport(nn.Module):
    def __init__(self, num_iter, inf=1e12):
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iter = num_iter
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.)))
        self.inf = inf

    def forward(self, scores, row_masks=None, col_masks=None):
        r"""
        Optimal transport with Sinkhorn.

        :param scores: torch.Tensor (B, M, N)
        :param row_masks: torch.Tensor (B, M)
        :param col_masks: torch.Tensor (B, N)
        :return matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape

        if row_masks is None:
            row_masks = torch.ones(batch_size, num_row, dtype=torch.bool).cuda()
        if col_masks is None:
            col_masks = torch.ones(batch_size, num_col, dtype=torch.bool).cuda()

        padded_row_masks = torch.zeros(batch_size, num_row + 1, dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(batch_size, num_col + 1, dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores[padded_score_masks] = -self.inf

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(batch_size, num_row + 1).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = torch.empty(batch_size, num_col + 1).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf

        outputs = log_sinkhorn_normalization(padded_scores, log_mu, log_nu, self.num_iter)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iter={})'.format(self.num_iter)
        return format_string


# class LearnableLogOptimalTransport(nn.Module):
#     def __init__(self, num_iter, inf=1e12):
#         super(LearnableLogOptimalTransport, self).__init__()
#         self.num_iter = num_iter
#         self.inf = inf
#         self.register_parameter('alpha', nn.Parameter(torch.tensor(1.)))
#
#     def forward(self, scores, row_masks=None, col_masks=None):
#         r"""
#         Optimal transport with Sinkhorn.
#
#         :param scores: torch.Tensor (B, M, N)
#         :param row_masks: torch.Tensor (B, M)
#         :param col_masks: torch.Tensor (B, N)
#         :return matching_scores: torch.Tensor (B, M+1, N+1)
#         """
#         batch_size, num_row, num_col = scores.shape
#
#         padded_col = self.alpha.expand(batch_size, num_row, 1)
#         padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
#         expanded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
#
#         norm = -math.log(num_row + num_col)
#
#         log_mu = torch.empty(num_row + 1).cuda()
#         log_mu[:num_row] = norm
#         log_mu[num_row] = math.log(num_col) + norm
#         log_mu = log_mu.expand(batch_size, num_row + 1)
#
#         log_nu = torch.empty(num_col + 1).cuda()
#         log_nu[:num_col] = norm
#         log_nu[num_col] = math.log(num_row) + norm
#         log_nu = log_nu.expand(batch_size, num_col + 1)
#
#         outputs = log_sinkhorn_normalization(expanded_scores, log_mu, log_nu, self.num_iter)
#         outputs = outputs - norm
#
#         return outputs
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '(num_iter={})'.format(self.num_iter)
#         return format_string


class LogOptimalTransport(nn.Module):
    def __init__(self, num_iter):
        super(LogOptimalTransport, self).__init__()
        self.num_iter = num_iter

    def forward(self, scores):
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        expanded_scores = zero_pad(scores.unsqueeze(1)).squeeze(1)
        for i in range(self.num_iter):
            expanded_scores = torch.cat([
                expanded_scores[:, :-1, :] - torch.logsumexp(expanded_scores[:, :-1, :], dim=2, keepdim=True),
                expanded_scores[:, -1:, :]
            ], dim=1)
            expanded_scores = torch.cat([
                expanded_scores[:, :, :-1] - torch.logsumexp(expanded_scores[:, :, :-1], dim=1, keepdim=True),
                expanded_scores[:, :, -1:]
            ], dim=2)
        expanded_scores = expanded_scores[:, :-1, :-1]
        return expanded_scores

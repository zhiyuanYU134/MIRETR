import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import correspondence_softnms


class CorrespondenceSoftNMS(nn.Module):
    def __init__(self, delta, sigma, joint_nms=True):
        super(CorrespondenceSoftNMS, self).__init__()
        self.delta = delta
        self.sigma = sigma
        self.joint_nms = joint_nms

    def forward(self, src_points, tgt_points, scores):
        suppressed_scores = correspondence_softnms(
            src_points, tgt_points, scores, delta=self.delta, sigma=self.sigma, joint_nms=self.joint_nms
        )

        return suppressed_scores

    def __repr__(self):
        format_string = self.__class__.__name__ + '(delta={:.2f}, sigma={:.2f}, joint={})'.format(
            self.delta, self.sigma, self.joint_nms
        )
        return format_string

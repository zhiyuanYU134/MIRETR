import torch

from vision3d.utils.point_cloud_utils import pairwise_distance
from ..nms.functional import nms


def correspondence_softnms(src_points, tgt_points, scores, delta=0.2, sigma=0.2, joint_nms=True):
    src_dist_map = pairwise_distance(src_points, src_points, clamp=True)
    tgt_dist_map = pairwise_distance(tgt_points, tgt_points, clamp=True)

    if joint_nms:
        dist_map = src_dist_map + tgt_dist_map
        factor_map = torch.maximum(1 - dist_map / (2 * delta) ** 2, torch.zeros_like(dist_map))
        factor_map = torch.exp(-factor_map ** 2 / sigma)
    else:
        src_factor_map = torch.maximum(1 - src_dist_map / delta ** 2, torch.zeros_like(src_dist_map))
        src_factor_map = torch.exp(-src_factor_map ** 2 / sigma)
        tgt_factor_map = torch.maximum(1 - tgt_dist_map / delta ** 2, torch.zeros_like(tgt_dist_map))
        tgt_factor_map = torch.exp(-tgt_factor_map ** 2 / sigma)
        factor_map = src_factor_map * tgt_factor_map

    indices = torch.arange(scores.shape[0]).cuda()
    factor_map[indices, indices] = 0.

    suppressed_scores = nms(scores, factor_map)

    return suppressed_scores

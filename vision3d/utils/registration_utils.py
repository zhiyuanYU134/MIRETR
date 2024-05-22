from cv2 import transform
import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .point_cloud_utils import (
    get_rotation_translation_from_transform, apply_transform, get_nearest_neighbor, get_point_to_node, pairwise_distance
)
from .torch_utils import index_select
import open3d as o3d



def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor
def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd
# Metrics

def compute_relative_rotation_error(gt_rotation, est_rotation):
    r"""
    [PyTorch/Numpy] Compute the isotropic Relative Rotation Error.
    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    :param gt_rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :param est_rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :return rre: torch.Tensor () or float
    """
    if isinstance(gt_rotation, torch.Tensor):
        x = 0.5 * (torch.trace(torch.matmul(est_rotation.T, gt_rotation)) - 1.)
        x = torch.clip(x, -1., 1.)
        x = torch.arccos(x)
    else:
        x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.)
        x = np.clip(x, -1., 1.)
        x = np.arccos(x)
    rre = 180. * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation, est_translation):
    r"""
    [Pytorch/Numpy] Compute the isotropic Relative Translation Error.
    RTE = \lVert t - \bar{t} \rVert_2

    :param gt_translation: torch.Tensor (3,) or numpy.ndarray (3,)
    :param est_translation: torch.Tensor (3,) or numpy.ndarray (3,)
    :return rte: torch.Tensor () or float
    """
    if isinstance(gt_translation, torch.Tensor):
        rte = torch.linalg.norm(gt_translation - est_translation)
    else:
        rte = np.linalg.norm(gt_translation - est_translation)
    return rte


def compute_registration_error(gt_transform, est_transform):
    r"""
    [PyTorch/Numpy] Compute the isotropic Relative Rotation Error and Relative Translation Error

    :param gt_transform: torch.Tensor (4, 4) or numpy.ndarray (4, 4)
    :param est_transform: numpy.ndarray (4, 4)
    :return rre: float
    :return rte: float
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte

def compute_add_error(gt_transform, est_transform,src_points):
    r"""
    [PyTorch/Numpy] Compute the isotropic Relative Rotation Error and Relative Translation Error

    :param gt_transform: torch.Tensor (M,4, 4) 
    :param est_transform: numpy.ndarray (N,4, 4)
    :param src_points: torch.Tensor (n,3) 
    :return rre: float
    :return rte: float
    """
    src_R=pairwise_distance(src_points,src_points).max()
    src_R=torch.sqrt(src_R)
    gt_src_points = apply_transform(src_points.unsqueeze(0), gt_transform)# (M,n,3) 
    est_src_points = apply_transform(src_points.unsqueeze(0), est_transform)# (N,n,3) 
    if gt_transform.is_cuda:
        add_metric=torch.zeros((len(gt_transform),len(est_transform))).cuda()
    else:
        add_metric=torch.zeros((len(gt_transform),len(est_transform)))
    for  i in range(len(gt_transform)):
        gt_src_point=gt_src_points[i]
        gt_src_point=gt_src_point.unsqueeze(0)# (1,n,3) 
        add_metric[i] = torch.sqrt((gt_src_point-est_src_points).abs().pow(2).sum(-1)).mean(-1)# (N) 
    return add_metric/src_R

def compute_adds_error(gt_transform, est_transform,src_points):
    r"""
    [PyTorch/Numpy] Compute the isotropic Relative Rotation Error and Relative Translation Error

    :param gt_transform: torch.Tensor (M,4, 4) 
    :param est_transform: numpy.ndarray (N,4, 4)
    :param src_points: torch.Tensor (n,3) 
    :return rre: float
    :return rte: float
    """
    src_R=pairwise_distance(src_points,src_points).max()
    src_R=torch.sqrt(src_R)
    print(src_R)
    gt_src_points = apply_transform(src_points.unsqueeze(0), gt_transform)# (M,n,3) 
    est_src_points = apply_transform(src_points.unsqueeze(0), est_transform)# (N,n,3) 
    if gt_transform.is_cuda:
        add_metric=torch.zeros((len(gt_transform),len(est_transform))).cuda()
    else:
        add_metric=torch.zeros((len(gt_transform),len(est_transform)))
    for  i in range(len(gt_transform)):
        gt_src_point=gt_src_points[i]
        gt_src_point=gt_src_point.unsqueeze(0).expand(len(est_src_points), len(src_points), 3)# (N,n,3) 
        dist2=torch.sqrt(pairwise_distance(est_src_points,gt_src_point))# (N,n,n) 
        dist2=dist2.min(-1)[0]# (N,n) 
        add_metric[i] = dist2.mean(-1)# (N) 
    return add_metric/src_R


def compute_rotation_mse_and_mae(gt_rotation, est_rotation):
    r"""
    [Numpy] Compute anisotropic rotation error (MSE and MAE).
    """
    gt_euler_angles = Rotation.from_dcm(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
    est_euler_angles = Rotation.from_dcm(est_rotation).as_euler('xyz', degrees=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae


def compute_translation_mse_and_mae(gt_translation, est_translation):
    r"""
    [Numpy] Compute anisotropic translation error (MSE and MAE).
    """
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae


def compute_transform_mse_and_mae(gt_transform, est_transform):
    r"""
    [Numpy] Compute anisotropic rotation and translation error (MSE and MAE).
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def compute_modified_chamfer_distance(raw_points, ref_points, src_points, gt_transform, est_transform):
    r"""
    [Numpy] Compute the modified chamfer distance.
    """
    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = pairwise_distance(aligned_src_points, raw_points).min(1).mean()
    # Q -> P_raw
    composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = pairwise_distance(ref_points, aligned_raw_points).min(1).mean()
    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance


def compute_overlap(ref_points, src_points, transform, positive_radius=0.1):
    r"""
    [Numpy] Compute the overlap of two point clouds.
    """
    src_points = apply_transform(src_points, transform)
    dist = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(dist < positive_radius)
    return overlap


def compute_inlier_ratio(ref_points, src_points, transform, positive_radius=0.1):
    r"""
    [Numpy] Computing the inlier ratio between a set of correspondences.
    """
    src_points = apply_transform(src_points, transform)
    distances = np.sqrt(((ref_points - src_points) ** 2).sum(1))
    inlier_ratio = np.mean(distances < positive_radius)
    return inlier_ratio


def compute_mean_distance(ref_points, src_points, transform):
    r"""
    [Numpy] Computing the mean distance between a set of correspondences.
    """
    src_points = apply_transform(src_points, transform)
    distances = np.sqrt(((ref_points - src_points) ** 2).sum(1))
    mean_distance = np.mean(distances)
    return mean_distance


# Ground Truth Utilities

def get_corr_indices(ref_points, src_points, transform, matching_radius):
    r"""
    [Numpy] Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    correspondences = np.array([(i, j) for i, indices in enumerate(indices_list) for j in indices], dtype=np.long)
    return correspondences


@torch.no_grad()
def get_node_corr_indices_and_counts(
        gt_corr_indices,
        ref_points,
        ref_nodes,
        src_points,
        src_nodes,
        ref_point_to_node=None,
        src_point_to_node=None,
        ref_node_sizes=None,
        src_node_sizes=None,
        return_scores=False
):
    r"""
    [PyTorch] Generate patch correspondences from point correspondences and the number of point correspondences within
    each patch correspondences.

    For each point correspondence, convert it to patch correspondence by replacing the point indices to the
    corresponding patch indices.

    We also define the proxy score for each patch correspondence as a estimation of the overlap ratio:
    s = (#point_corr / #point_in_ref_patch + #point_corr / #point_in_src_patch) / 2

    :param gt_corr_indices: point correspondences
    :param ref_points: reference point cloud
    :param ref_nodes: reference patch points
    :param src_points: source point cloud
    :param src_nodes: source patch points
    :param ref_point_to_node: point-to-node mapping for the reference point cloud
    :param src_point_to_node: point-to-node mapping for the source point cloud
    :param ref_node_sizes: the number of points in each patch of the reference point cloud
    :param src_node_sizes: the number of points in each patch of the source point cloud
    :param return_scores: whether return the proxy score for each patch correspondences
    """
    if ref_point_to_node is None or (return_scores and ref_node_sizes is None):
        if return_scores:
            ref_point_to_node, ref_node_sizes = get_point_to_node(ref_points, ref_nodes, return_counts=True)
        else:
            ref_point_to_node = get_point_to_node(ref_points, ref_nodes)

    if src_point_to_node is None or (return_scores and src_node_sizes is None):
        if return_scores:
            src_point_to_node, src_node_sizes = get_point_to_node(src_points, src_nodes, return_counts=True)
        else:
            src_point_to_node = get_point_to_node(src_points, src_nodes)

    src_length_node = src_nodes.shape[0]
    ref_corr_indices = gt_corr_indices[:, 0]
    src_corr_indices = gt_corr_indices[:, 1]

    ref_node_corr_indices = ref_point_to_node[ref_corr_indices]
    src_node_corr_indices = src_point_to_node[src_corr_indices]
    node_corr_indices = ref_node_corr_indices * src_length_node + src_node_corr_indices
    node_corr_indices, node_corr_counts = torch.unique(node_corr_indices, return_counts=True)
    ref_node_corr_indices = node_corr_indices // src_length_node
    src_node_corr_indices = node_corr_indices % src_length_node
    node_corr_indices = torch.stack([ref_node_corr_indices, src_node_corr_indices], dim=1)

    if return_scores:
        ref_node_corr_scores = node_corr_counts / ref_node_sizes[ref_node_corr_indices]
        src_node_corr_scores = node_corr_counts / src_node_sizes[src_node_corr_indices]
        node_corr_scores = (ref_node_corr_scores + src_node_corr_scores) / 2
        return node_corr_indices, node_corr_counts, node_corr_scores
    else:
        return node_corr_indices, node_corr_counts


@torch.no_grad()
def cal_R(ref_node_corr_knn_points,ref_node_corr_knn_masks,patch_list):   
    points=ref_node_corr_knn_points[patch_list[0]][ref_node_corr_knn_masks[patch_list[0]]]
    for i in range(len(patch_list)-1):
        points=torch.cat([points,ref_node_corr_knn_points[patch_list[i+1]][ref_node_corr_knn_masks[patch_list[i+1]]]],dim=0)
    dist=pairwise_distance(points,points)       
    return dist.max()



@torch.no_grad()
def get_node_corr_indices_and_overlaps(
        ref_nodes,
        src_nodes,
        ref_knn_points,
        src_knn_points,
        transform,
        pos_radius,
        point_pos_radius,
        ref_masks=None,
        src_masks=None,
        ref_knn_masks=None,
        src_knn_masks=None
):
    r"""
    Generate ground truth node correspondences.
    Each node is composed of its k nearest points. A pair of points match if the distance between them is below
    `self.pos_radius`.

    :param ref_nodes: torch.Tensor (M, 3)
    :param src_nodes: torch.Tensor (N, 3)
    :param ref_knn_points: torch.Tensor (M, K, 3)
    :param src_knn_points: torch.Tensor (N, K, 3)
    :param transform: torch.Tensor (4, 4)
    :param pos_radius: float
    :param ref_masks: torch.BoolTensor (M,) (default: None)
    :param src_masks: torch.BoolTensor (N,) (default: None)
    :param ref_knn_masks: torch.BoolTensor (M, K) (default: None)
    :param src_knn_masks: torch.BoolTensor (N, K) (default: None)

    :return corr_indices: torch.LongTensor (num_corr, 2)
    :return corr_overlaps: torch.Tensor (num_corr,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    if ref_masks is not None and src_masks is not None:
        node_masks = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))
    else:
        node_masks = None

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_distances = torch.sqrt(((ref_knn_points - ref_nodes.unsqueeze(1)) ** 2).sum(-1))  # (M, K)
    if ref_knn_masks is not None:
        ref_knn_distances[~ref_knn_masks] = 0.
    ref_radius = ref_knn_distances.max(1)[0]  # (M,)
    src_knn_distances = torch.sqrt(((src_knn_points - src_nodes.unsqueeze(1)) ** 2).sum(-1))  # (N, K)
    if src_knn_masks is not None:
        src_knn_distances[~src_knn_masks] = 0.
    src_radius = src_knn_distances.max(1)[0]  # (N,)
    dist_map = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    masks = torch.gt(ref_radius.unsqueeze(1) + src_radius.unsqueeze(0) + pos_radius - dist_map, 0)  # (M, N)
    if node_masks is not None:
        masks = torch.logical_and(masks, node_masks)
    ref_indices, src_indices = torch.nonzero(masks, as_tuple=True)

    if ref_knn_masks is not None and src_knn_masks is not None:
        ref_knn_masks = ref_knn_masks[ref_indices]  # (B, K)
        src_knn_masks = src_knn_masks[src_indices]  # (B, K)
        node_knn_masks = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))
    else:
        node_knn_masks = None

    # compute overlaps
    ref_knn_points = ref_knn_points[ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[src_indices]  # (B, K, 3)
    dist_map = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    if node_knn_masks is not None:
        dist_map[~node_knn_masks] = 1e12
    point_corr_map = torch.lt(dist_map, point_pos_radius ** 2)
    ref_overlap_counts = torch.count_nonzero(point_corr_map.sum(-1), dim=-1).float()
    src_overlap_counts = torch.count_nonzero(point_corr_map.sum(-2), dim=-1).float()
    if node_knn_masks is not None:
        ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()
        src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()
    else:
        ref_overlaps = ref_overlap_counts / ref_knn_points.shape[1]  # (B,)
        src_overlaps = src_overlap_counts / src_knn_points.shape[1]  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    masks = torch.gt(overlaps, 0.05)
    ref_corr_indices = ref_indices[masks]
    src_corr_indices = src_indices[masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[masks]

    return corr_indices, corr_overlaps


@torch.no_grad()
def get_corr_indices_and_distances_from_node_corr_indices(
        ref_knn_points,
        src_knn_points,
        ref_knn_indices,
        src_knn_indices,
        gt_node_corr_indices,
        transform,
        matching_radius,
        ref_knn_masks=None,
        src_knn_masks=None,
        return_distance=False
):
    src_knn_points = apply_transform(src_knn_points, transform)
    gt_ref_corr_indices = gt_node_corr_indices[:, 0]  # (P,)
    gt_src_corr_indices = gt_node_corr_indices[:, 1]  # (P,)
    ref_node_corr_knn_indices = ref_knn_indices[gt_ref_corr_indices]  # (P, K)
    src_node_corr_knn_indices = src_knn_indices[gt_src_corr_indices]  # (P, K)
    ref_node_corr_knn_points = ref_knn_points[gt_ref_corr_indices]  # (P, K, 3)
    src_node_corr_knn_points = src_knn_points[gt_src_corr_indices]  # (P, K, 3)
    dist_map = torch.sqrt(pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points))  # (P, K, K)
    pos_masks = torch.lt(dist_map, matching_radius)
    if ref_knn_masks is not None and src_knn_masks is not None:
        ref_node_corr_knn_masks = ref_knn_masks[gt_ref_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_knn_masks[gt_src_corr_indices]  # (P, K)
        corr_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        pos_masks = torch.logical_and(pos_masks, corr_masks)  # (P, K, K)
    sel_corr_indices, sel_ref_indices, sel_src_indices = torch.nonzero(pos_masks, as_tuple=True)  # (C,) (C,) (C,)
    ref_corr_indices = ref_node_corr_knn_indices[sel_corr_indices, sel_ref_indices]
    src_corr_indices = src_node_corr_knn_indices[sel_corr_indices, sel_src_indices]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    if return_distance:
        corr_distances = dist_map[sel_corr_indices, sel_ref_indices, sel_src_indices]
        return corr_indices, corr_distances
    else:
        return corr_indices


@torch.no_grad()
def get_node_non_overlap_ratio(
        ref_points,
        src_points,
        ref_knn_points,
        src_knn_points,
        ref_knn_indices,
        src_knn_indices,
        gt_node_corr_indices,
        transform,
        matching_radius,
        ref_knn_masks,
        src_knn_masks,
        eps=1e-5
):
    gt_corr_indices = get_corr_indices_and_distances_from_node_corr_indices(
        ref_knn_points, src_knn_points, ref_knn_indices, src_knn_indices, gt_node_corr_indices, transform,
        matching_radius, ref_knn_masks=None, src_knn_masks=None
    )
    unique_ref_corr_indices = torch.unique(gt_corr_indices[:, 0])
    unique_src_corr_indices = torch.unique(gt_corr_indices[:, 1])
    ref_overlap_masks = torch.zeros(ref_points.shape[0] + 1).cuda()
    src_overlap_masks = torch.zeros(src_points.shape[0] + 1).cuda()
    ref_overlap_masks.index_fill_(0, unique_ref_corr_indices, 1.)
    src_overlap_masks.index_fill_(0, unique_src_corr_indices, 1.)
    ref_knn_overlap_masks = index_select(ref_overlap_masks, ref_knn_indices, dim=0)
    src_knn_overlap_masks = index_select(src_overlap_masks, src_knn_indices, dim=0)
    ref_knn_overlap_ratio = (ref_knn_overlap_masks * ref_knn_masks).sum(1) / (ref_knn_masks.sum(1) + eps)
    src_knn_overlap_ratio = (src_knn_overlap_masks * src_knn_masks).sum(1) / (src_knn_masks.sum(1) + eps)
    ref_knn_non_overlap_ratio = 1. - ref_knn_overlap_ratio
    src_knn_non_overlap_ratio = 1. - src_knn_overlap_ratio
    return ref_knn_non_overlap_ratio, src_knn_non_overlap_ratio


# Matching Utilities

@torch.no_grad()
def extract_corr_indices_from_scores(score_map, mutual=False, use_slack=False, threshold=0.):
    r"""
    [PyTorch] Extract the indices of correspondences from matching scores matrix.

    :param score_map: torch.Tensor (num_point0, num_point1) or (num_point0 + 1, num_point1 + 1) according to `slack`
        `scores` is the logarithmic matching probabilities.
    :param mutual: bool (default: False), whether to get mutual correspondences
    :param use_slack: bool (default: False), whether to use slack variables
    :param threshold: float (default: 0)

    :return src_corr_indices: torch.LongTensor (num_corr,)
    :return tgt_corr_indices: torch.LongTensor (num_corr,)
    """
    score_map = torch.exp(score_map)
    ref_length, src_length = score_map.shape

    ref_max_scores, ref_max_indices = torch.max(score_map, dim=1)
    ref_indices = torch.arange(ref_length).cuda()
    ref_score_map = torch.zeros_like(score_map)
    ref_score_map[ref_indices, ref_max_indices] = ref_max_scores
    ref_corr_map = torch.gt(ref_score_map, threshold)

    src_max_scores, src_max_indices = torch.max(score_map, dim=0)
    src_indices = torch.arange(src_length).cuda()
    src_score_map = torch.zeros_like(score_map)
    src_score_map[src_max_indices, src_indices] = src_max_scores
    src_corr_map = torch.gt(src_score_map, threshold)

    if use_slack:
        ref_corr_map = ref_corr_map[:-1, :-1]
        src_corr_map = src_corr_map[:-1, :-1]

    if mutual:
        corr_map = torch.logical_and(ref_corr_map, src_corr_map)
    else:
        corr_map = torch.logical_or(ref_corr_map, src_corr_map)

    ref_corr_indices, src_corr_indices = torch.nonzero(corr_map, as_tuple=True)

    return ref_corr_indices, src_corr_indices


def extract_corr_indices_from_feats(ref_feats, src_feats, mutual=False):
    r"""
    [PyTorch/Numpy] Extract correspondence indices from features.
    """
    if isinstance(ref_feats, torch.Tensor):
        feat_dists = pairwise_distance(ref_feats, src_feats)
        ref_corr_indices, src_corr_indices = extract_corr_indices_from_scores(
            -feat_dists, mutual=mutual, use_slack=False, threshold=0.
        )
        return ref_corr_indices, src_corr_indices
    else:
        ref_nn_indices = get_nearest_neighbor(ref_feats, src_feats, return_index=True)[1]
        if mutual:
            src_nn_indices = get_nearest_neighbor(src_feats, ref_feats, return_index=True)[1]
            ref_indices = np.arange(ref_feats.shape[0])
            ref_masks = np.equal(src_nn_indices[ref_nn_indices], ref_indices)
            ref_corr_indices = ref_indices[ref_masks]
            src_corr_indices = ref_nn_indices[ref_corr_indices]
        else:
            ref_corr_indices = np.arange(ref_feats.shape[0])
            src_corr_indices = ref_nn_indices
            # src_indices = np.arange(src_feats.shape[0])
            # ref_corr_indices = np.concatenate([ref_indices, src_nn_indices], axis=0)
            # src_corr_indices = np.concatenate([ref_nn_indices, src_indices], axis=0)
        return ref_corr_indices, src_corr_indices


def extract_correspondences_from_feats(
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        mutual=False,
        return_indices=False,
        return_feat_dist=False
):
    r"""
    [Pytorch/Numpy] Extract correspondences from features.
    """
    ref_corr_indices, src_corr_indices = extract_corr_indices_from_feats(ref_feats, src_feats, mutual=mutual)

    outputs = []
    if return_indices:
        outputs.append(ref_corr_indices)
        outputs.append(src_corr_indices)
    else:
        outputs.append(ref_points[ref_corr_indices])
        outputs.append(src_points[src_corr_indices])
    if return_feat_dist:
        ref_feats = ref_feats[ref_corr_indices]
        src_feats = src_feats[src_corr_indices]
        if isinstance(ref_feats, torch.Tensor):
            feat_dists = torch.linalg.norm(ref_feats - src_feats, dim=1)
        else:
            feat_dists = np.linalg.norm(ref_feats - src_feats, axis=1)
        outputs.append(feat_dists)
    return outputs


def extract_correspondences_from_scores(
        ref_points,
        src_points,
        scores,
        mutual=False,
        use_slack=False,
        threshold=0.
):
    r"""
    [PyTorch] Extract correspondences from matching scores matrix.

    :param ref_points: torch.Tensor (num_point0, 3)
    :param src_points: torch.Tensor (num_point1, 3)
    :param scores: torch.Tensor (num_point0, num_point1) or (num_point0 + 1, num_point1 + 1) according to `slack`
        `scores` is the logarithmic matching probabilities.
    :param mutual: bool (default: False), whether to get mutual correspondences
    :param use_slack: bool (default: False), whether to use slack variables
    :param threshold: float (default: 0)
    """
    ref_corr_indices, src_corr_indices = extract_corr_indices_from_scores(
        scores, mutual=mutual, use_slack=use_slack, threshold=threshold
    )
    return ref_points[ref_corr_indices], src_points[src_corr_indices]


def extract_correspondences_from_scores_threshold(
        ref_points,
        src_points,
        scores,
        threshold,
        use_slack=False,
        return_indices=False
):
    scores = torch.exp(scores)
    if use_slack:
        scores = scores[:-1, :-1]
    masks = torch.gt(scores, threshold)
    corr_indices0, corr_indices1 = torch.nonzero(masks, as_tuple=True)

    if return_indices:
        return corr_indices0, corr_indices1
    else:
        return ref_points[corr_indices0], src_points[corr_indices1]


def extract_correspondences_from_scores_topk(
        ref_points,
        src_points,
        scores,
        k,
        use_slack=False,
        largest=True,
        return_indices=False
):
    corr_indices = scores.flatten().topk(k=k, largest=largest)[1]
    ref_corr_indices = corr_indices // scores.shape[1]
    src_corr_indices = corr_indices % scores.shape[1]
    if use_slack:
        masks = (ref_corr_indices != scores.shape[0] - 1) & (src_corr_indices != scores.shape[1] - 1)
        ref_corr_indices = ref_corr_indices[masks]
        src_corr_indices = src_corr_indices[masks]

    if return_indices:
        return ref_corr_indices, src_corr_indices
    else:
        return ref_points[ref_corr_indices], src_points[src_corr_indices]


def extract_mutual_correspondences_from_scores(
        ref_points,
        src_points,
        scores,
        slack=False,
        matching_threshold=0.2
):
    r"""
    [PyTorch] Get mutual correspondences from matching scores matrix.
    Deprecated. Use `get_correspondences_from_scores` with `mutual=True` instead.
    """
    print('Warning: "extract_mutual_correspondences_from_scores" is deprecated. \
           Use "extract_correspondences_from_scores" with "mutual=True" instead.')
    if not slack:
        ref_max_scores, ref_max_indices = scores.max(dim=1)
        src_max_indices = scores.argmax(dim=0)
        ref_indices = torch.arange(ref_max_scores.shape[0]).cuda()
        ref_masks = (src_max_indices[ref_max_indices] == ref_indices) & (ref_max_scores > matching_threshold)
        ref_corr_indices = ref_indices[ref_masks]
        src_corr_indices = ref_max_indices[ref_masks]
        return ref_points[ref_corr_indices], src_points[src_corr_indices]
    else:
        ref_max_scores, ref_max_indices = scores[:-1, ].max(dim=1)
        src_max_indices = scores.argmax(dim=0)
        ref_indices = torch.arange(ref_max_scores.shape[0]).cuda()
        ref_masks = (ref_max_indices != scores.shape[1] - 1) & (src_max_indices[ref_max_indices] == ref_indices)
        ref_masks = ref_masks & (ref_max_scores.exp() > matching_threshold)
        ref_corr_indices = ref_indices[ref_masks]
        src_corr_indices = ref_max_indices[ref_masks]
        return ref_points[ref_corr_indices], src_points[src_corr_indices]


# Evaluation Utilities

def evaluate_correspondences_from_feats(
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        transform,
        positive_radius=0.1,
        mutual=False
):
    print('Warning: "evaluate_correspondences_from_feats" is deprecated. \
           Use "extract_correspondences_from_feats" and "evaluate_correspondences" instead.')

    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=positive_radius)

    ref_corr_points, src_corr_points = extract_correspondences_from_feats(
        ref_points, src_points, ref_feats, src_feats, mutual=mutual
    )

    inlier_ratio = compute_inlier_ratio(ref_corr_points, src_corr_points, transform, positive_radius=positive_radius)
    mean_distance = compute_mean_distance(ref_corr_points, src_corr_points, transform)

    result_dict = {
        'overlap': overlap,
        'inlier_ratio': inlier_ratio,
        'mean_dist': mean_distance,
        'num_corr': ref_points.shape[0]
    }
    return result_dict


def evaluate_correspondences(ref_points, src_points, transform, positive_radius=0.1):
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=positive_radius)
    inlier_ratio = compute_inlier_ratio(ref_points, src_points, transform, positive_radius=positive_radius)
    mean_distance = compute_mean_distance(ref_points, src_points, transform)
    result_dict = {
        'overlap': overlap,
        'inlier_ratio': inlier_ratio,
        'mean_dist': mean_distance,
        'num_corr': ref_points.shape[0]
    }
    return result_dict

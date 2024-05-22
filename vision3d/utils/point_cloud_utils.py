import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.linalg import expm, norm
import time
eps = 1e-8

""" def get_point2trans_index(ref_points,src_points,trans,dist_thre):
    point2trans_index=torch.full([len(ref_points)],-1).cuda()
    for i in range(len(trans)):
        tran=trans[i]
        src_points_tran=apply_transform(src_points,tran)
        dist= pairwise_distance(ref_points, src_points_tran)
        dist=torch.lt(dist,dist_thre*dist_thre).float()
        dist=dist.sum(-1)
        dist=torch.gt(dist,0)
        point2trans_index[dist]=i
        

    return point2trans_index """

def get_point2trans_index(ref_points,src_points,trans,dist_thre):
    point2trans_index=torch.full([len(ref_points)],-1).cuda()
    min_dists=torch.ones([len(trans),len(ref_points)]).cuda()
    for i in range(len(trans)):
        tran=trans[i]
        src_points_tran=apply_transform(src_points,tran)
        dist= pairwise_distance(ref_points, src_points_tran)
        min_dists[i]=torch.min(dist,dim=1)[0]
    min_dist,min_dists_indices=torch.min(min_dists,dim=0)
    min_dist_mask=torch.lt(min_dist,dist_thre*dist_thre)
    point2trans_index[min_dist_mask]=min_dists_indices[min_dist_mask]
    return point2trans_index

def get_corr2trans_index(ref_points,src_points,est_trans,dist_thre):
    corr2trans_index=torch.full([len(ref_points)],-1)
    aligned_src_corr_points = apply_transform(src_points.unsqueeze(0), est_trans)
    min_dists = torch.sum((ref_points.unsqueeze(0) - aligned_src_corr_points) ** 2, dim=2)
    min_dist,min_dists_indices=torch.min(min_dists,dim=0)
    min_dist_mask=torch.lt(min_dist,dist_thre*dist_thre)
    print(torch.sum(min_dist_mask.float())*100/len(ref_points))
    corr2trans_index[min_dist_mask]=min_dists_indices[min_dist_mask]
    return corr2trans_index

def get_sym_corr2trans_index(ref_points,src_points,est_trans,dist_thre):
    corr2trans_index=torch.full([len(ref_points)],-1)
    aligned_src_corr_points = apply_transform(src_points.unsqueeze(0), est_trans)
    min_dists = torch.sum((ref_points.unsqueeze(0) - aligned_src_corr_points) ** 2, dim=2)
    min_dist,min_dists_indices=torch.min(min_dists,dim=0)
    min_dist_mask=torch.lt(min_dist,dist_thre*dist_thre)
    corr2trans_index[min_dist_mask]=min_dists_indices[min_dist_mask]
    return corr2trans_index

def get_compatibility(src_keypts, tgt_keypts, with_dist_corr_compatibility=False, dist_sigma=None):
    # # (num_proposal, max_point,3), (num_proposal, max_point,3)
    #################################
    # Step1: corr_compatibily
    #################################
    src_dist = torch.norm((src_keypts.unsqueeze(-2) - src_keypts.unsqueeze(-3)), dim=-1) # [n1, n1]
    tgt_dist = torch.norm((tgt_keypts.unsqueeze(-2) - tgt_keypts.unsqueeze(-3)), dim=-1) # [n1, n1]
    
    corr_compatibility = torch.min(torch.stack((src_dist/(tgt_dist+1e-12),tgt_dist/(src_dist+1e-12)), dim=-1), dim=-1).values + torch.eye(src_dist.shape[-1]).to(src_dist)
    corr_compatibility = torch.clamp((corr_compatibility / 0.9)**2, min=0)
    
    #################################
    # Step2: dist_compatibily
    #################################
    if with_dist_corr_compatibility:
        dist_corr_compatibility = (src_dist + tgt_dist)/2
        dist_corr_compatibility = torch.clamp(1.0 - dist_corr_compatibility ** 2 / dist_sigma ** 2, min=0) # [n1, n1]
        corr_compatibility *= dist_corr_compatibility
        corr_compatibility /= corr_compatibility.max(dim=-1, keepdim=True).values.max(dim=-1, keepdim=True).values
    return corr_compatibility

# Basic Utilities
def is_rotation(R):
    assert R.shape == (
        3,
        3,
    ), f"rotation matrix should be in shape (3, 3) but got {R.shape} input."
    rrt = R @ R.t()
    I = torch.eye(3)
    err = torch.norm(I - rrt)
    return err < eps


def skew_symmetric(vectors):
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)

    r00 = torch.zeros_like(vectors[:, 0])
    r01 = -vectors[:, 2]
    r02 = vectors[:, 1]
    r10 = vectors[:, 2]
    r11 = torch.zeros_like(r00)
    r12 = -vectors[:, 0]
    r20 = -vectors[:, 1]
    r21 = vectors[:, 0]
    r22 = torch.zeros_like(r00)

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(
        -1, 3, 3
    )
    return R


def axis_angle_to_rotation(axis_angles):
    if axis_angles.dim() == 1:
        axis_angles = axis_angles.unsqueeze(0)

    angles = torch.norm(axis_angles, p=2, dim=-1, keepdim=True)
    axis = axis_angles / angles

    K = skew_symmetric(axis)
    K_square = torch.bmm(K, K)
    I = torch.eye(3).to(axis_angles.device).repeat(K.shape[0], 1, 1)

    R = (
        I
        + torch.sin(angles).unsqueeze(-1) * K
        + (1 - torch.cos(angles).unsqueeze(-1)) * K_square
    )

    return R.squeeze(0)

def rotation_to_axis_angle(R):
    if R.dim() == 2:
        R = R.unsqueeze(0)

    theta = torch.acos(((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2 + eps)
    sin_theta = torch.sin(theta)

    singular = torch.zeros(3, dtype=torch.float32).to(theta.device)

    multi = 1 / (2 * sin_theta + eps)
    rx = multi * (R[:, 2, 1] - R[:, 1, 2]) * theta
    ry = multi * (R[:, 0, 2] - R[:, 2, 0]) * theta
    rz = multi * (R[:, 1, 0] - R[:, 0, 1]) * theta

    axis_angles = torch.stack((rx, ry, rz), dim=-1)
    singular_indices = torch.logical_or(sin_theta == 0, sin_theta.isnan())
    axis_angles[singular_indices] = singular
    return axis_angles.squeeze(0),torch.logical_not(singular_indices)

def pairwise_distance_ori(points0, points1, normalized=False, clamp=False):
    r"""
    [PyTorch/Numpy] Pairwise distance of two point clouds.

    :param points0: torch.Tensor (d0, ..., dn, num_point0, num_feature)
    :param points1: torch.Tensor (d0, ..., dn, num_point1, num_feature)
    :param normalized: bool (default: False)
        If True, the points are normalized, so a2 and b2 both 1. This enables us to use 2 instead of a2 + b2 for
        simplicity.
    :param clamp: bool (default: False)
        If True, all value will be assured to be non-negative.
    :return: dist: torch.Tensor (d0, ..., dn, num_point0, num_point1)
    """
    if isinstance(points0, torch.Tensor):
        ab = torch.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = torch.sum(points0 ** 2, dim=-1).unsqueeze(-1)
            b2 = torch.sum(points1 ** 2, dim=-1).unsqueeze(-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = torch.maximum(dist2, torch.zeros_like(dist2))
    else:
        ab = np.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = np.expand_dims(np.sum(points0 ** 2, axis=-1), axis=-1)
            b2 = np.expand_dims(np.sum(points1 ** 2, axis=-1), axis=-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = np.maximum(dist2, np.zeros_like(dist2))
    return dist2

def pairwise_distance(points0, points1, normalized=False, clamp=False):
    if isinstance(points0, torch.Tensor):
        if len(points0.shape)==3:
            
            dist2 = (points0.unsqueeze(2)-points1.unsqueeze(1)).abs().pow(2).sum(-1)
        else:
            dist2 = (points0.unsqueeze(1)-points1.unsqueeze(0)).abs().pow(2).sum(-1)
    else:
        ab = np.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = np.expand_dims(np.sum(points0 ** 2, axis=-1), axis=-1)
            b2 = np.expand_dims(np.sum(points1 ** 2, axis=-1), axis=-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = np.maximum(dist2, np.zeros_like(dist2))
    return dist2

def cal_sim_sp(src_keypts, tgt_keypts, sigma_spat = 0.1):
    #(N,m,3)
    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)#(N,m,m)
    corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)#(N,m,m)
    corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / sigma_spat ** 2, min=0)#(N,m,m)
    return corr_compatibility

def get_nearest_neighbor(ref_points, src_points, return_index=False):
    r"""
    [PyTorch/Numpy] For each item in ref_points, find its nearest neighbor in src_points.

    The PyTorch implementation is based on pairwise distances, thus it cannot be used for large point clouds.
    """
    if isinstance(ref_points, torch.Tensor):
        distances = pairwise_distance(ref_points, src_points)
        nn_distances, nn_indices = distances.min(dim=1)
        if return_index:
            return nn_distances, nn_indices
        else:
            return nn_distances
    else:
        kd_tree1 = cKDTree(src_points)
        distances, indices = kd_tree1.query(ref_points, k=1, n_jobs=-1)
        if return_index:
            return distances, indices
        else:
            return distances


def get_point_to_node(points, nodes, return_counts=False):
    r"""
    [PyTorch/Numpy] Distribute points to the nearest node. Each point is distributed to only one node.

    :param points: torch.Tensor (num_point, num_channel)
    :param nodes: torch.Tensor (num_node, num_channel)
    :param return_counts: bool (default: False)
        If True, return the number of points in each node.
    :return: indices: torch.Tensor (num_point)
        The indices of the nodes to which the points are distributed.
    """
    if isinstance(points, torch.Tensor):
        """ print("isinstance") """
        distances = pairwise_distance(points, nodes)
        indices = distances.min(dim=1)[1]
        if return_counts:
            unique_indices, unique_counts = torch.unique(indices, return_counts=True)
            node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()
            node_sizes[unique_indices] = unique_counts
            return indices, node_sizes
        else:
            return indices
    else:
        """ print("instance") """
        _, indices = get_nearest_neighbor(points, nodes, return_index=True)
        if return_counts:
            unique_indices, unique_counts = np.unique(indices, return_counts=True)
            node_sizes = np.zeros(nodes.shape[0], dtype=np.int64)
            node_sizes[unique_indices] = unique_counts
            return indices, node_sizes
        else:
            return indices


def get_knn_indices(points, nodes, k, return_distance=False):
    r"""
    [PyTorch] Find the k nearest points for each node.

    :param points: torch.Tensor (num_point, num_channel)
    :param nodes: torch.Tensor (num_node, num_channel)
    :param k: int
    :param return_distance: bool
    :return knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    dists = pairwise_distance(nodes, points)
    knn_distances, knn_indices = dists.topk(dim=1, k=k, largest=False)
    if return_distance:
        return torch.sqrt(knn_distances), knn_indices
    else:
        return knn_indices


def get_point_indices_in_node(points, nodes, num_sample, point_to_node=None, sorted=False):
    r"""
    [PyTorch] Get the points for each node. If there are too many points, only the nearest `max_point` points are used.

    :param points: torch.Tensor (num_point, 3)
    :param nodes: torch.Tensor (num_node, 3)
    :param num_sample: int, the max number of points in each node
    :param point_to_node: torch.Tensor (num_point), the index of node each point belongs to (default: None)
    :param sorted: bool, sort the result or not
    :return node_knn_indices: the indices of points in each node (being num_point is masked).
    """
    num_point = points.shape[0]
    num_node = nodes.shape[0]
    if point_to_node is None:
        point_to_node = get_point_to_node(points, nodes)  # (num_point)
    node_knn_indices = get_knn_indices(points, nodes, num_sample)  # (num_node, max_point)
    node_indices = torch.arange(num_node).cuda().unsqueeze(1).expand(num_node, num_sample)
    masks = torch.ne(point_to_node[node_knn_indices], node_indices)
    node_knn_indices[masks] = num_point
    if sorted:
        node_knn_indices = torch.sort(node_knn_indices, dim=1, descending=False)
    return node_knn_indices
@torch.no_grad()
def farthest_point_sample(data,npoints):
    """
    Args:
        data:输入的tensor张量，排列顺序 N,D
        Npoints: 需要的采样点

    Returns:data->采样点集组成的tensor，每行是一个采样点
    """
    N,D = data.shape #N是点数，D是维度
    xyz = data[:,:3] #只需要坐标
    centroids = torch.zeros(size=(npoints,)).cuda() #最终的采样点index
    dictance = torch.ones(size=(N,)).cuda() *1e10 #距离列表,一开始设置的足够大,保证第一轮肯定能更新dictance
    farthest = torch.randint(low=0,high=N,size=(1,)).cuda()  #随机选一个采样点的index
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest,:]
        dict = ((xyz-centroid)**2).sum(dim=-1)
        mask = dict < dictance
        dictance[mask] = dict[mask]
        farthest = torch.argmax(dictance,dim=-1)

    #data= data[centroids.type(torch.long)]
    return centroids.type(torch.long)


@torch.no_grad()
def get_point_to_node_indices_and_masks(points, nodes, num_sample, return_counts=False):
    r"""
    [PyTorch] Perform point-to-node partition to the point cloud.

    :param points: torch.Tensor (num_point, 3)
    :param nodes: torch.Tensor (num_node, 3)
    :param num_sample: int
    :param return_counts: bool, whether to return `node_sizes`

    :return point_node_indices: torch.LongTensor (num_point,)
    :return node_sizes [Optional]: torch.LongTensor (num_node,)
    :return node_masks: torch.BoolTensor (num_node,)
    :return node_knn_indices: torch.LongTensor (num_node, max_point)
    :return node_knn_masks: torch.BoolTensor (num_node, max_point)
    """
    """ start_time=time.time() """

    point_to_node, node_sizes = get_point_to_node(points, nodes, return_counts=True)
    node_masks = torch.gt(node_sizes, 0)


    """ loading_time = time.time() - start_time
    print("get_point_to_node")
    print(loading_time) """

    node_knn_indices = get_knn_indices(points, nodes, num_sample)  # (num_node, max_point)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, min(num_sample,points.shape[0]))



    node_knn_masks = torch.eq(point_to_node[node_knn_indices], node_indices)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)


    if return_counts:
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks

@torch.no_grad()
def get_pointc_to_superpoint_indices_and_masks(points, nodes, num_sample, return_counts=False):
    r"""
    [PyTorch] Perform point-to-node partition to the point cloud.

    :param points: torch.Tensor (num_point, 3)
    :param nodes: torch.Tensor (num_node, 3)
    :param num_sample: int
    :param return_counts: bool, whether to return `node_sizes`

    :return point_node_indices: torch.LongTensor (num_point,)
    :return node_sizes [Optional]: torch.LongTensor (num_node,)
    :return node_masks: torch.BoolTensor (num_node,)
    :return node_knn_indices: torch.LongTensor (num_node, max_point)
    :return node_knn_masks: torch.BoolTensor (num_node, max_point)
    """
    """ start_time=time.time() """

    point_to_node, node_sizes = get_point_to_node(points, nodes, return_counts=True)
    node_masks = torch.gt(node_sizes, 0)


    """ loading_time = time.time() - start_time
    print("get_point_to_node")
    print(loading_time) """

    node_knn_indices = get_knn_indices(points, nodes, num_sample)  # (num_node, max_point)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, num_sample)
    node_knn_masks = torch.eq(point_to_node[node_knn_indices], node_indices)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)


    if return_counts:
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks

@torch.no_grad()
def get_ball_query_indices_and_masks(points, nodes, radius, num_sample, return_counts=False, eps=1e-5):
    node_knn_distances, node_knn_indices = get_knn_indices(points, nodes, num_sample, return_distance=True)
    node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])  # (N, k)
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)  # (N, k)

    if return_counts:
        node_sizes = node_knn_masks.sum(1)  # (N, 1)
        return node_knn_indices, node_knn_masks, node_sizes
    else:
        return node_knn_indices, node_knn_masks


# Transformation Utilities

def apply_transform(points, transform):
    r"""
    [PyTorch/Numpy] Apply a rigid transform to points.

    Given a point cloud P(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t.

    In the implementation, P is (N, 3), so R should be transposed.

    There are two cases supported:
    1. points is (d0, .., dn, 3), transform is (4, 4), the output points are (d0, ..., dn, 3).
       In this case, the transform is applied to all points.
    2. points is (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    :param points: torch.Tensor (d0, ..., dn, 3) or (B, N, 3)
    :param transform: torch.Tensor (4, 4) or (B, 4, 4)
    :return: points, torch.Tensor (d0, ..., dn, 3) or (B, N, 3)
    """
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = points @ rotation.transpose(-1, -2) + translation
        points = points.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = points @ rotation.transpose(-1, -2) + translation
    else:
        raise ValueError('Incompatible shapes between points {} and transform {}.'.format(
            tuple(points.shape), tuple(transform.shape)
        ))
    return points


def compose_transforms(transforms):
    r"""
    Compose transforms from the first one to the last one.
    T = T_{n_1} \circ T_{n_2} \circ ... \circ T_1 \circ T_0
    :param transforms: list of torch.Tensor [(4, 4)]
    :return transform: torch.Tensor (4, 4)
    """
    final_transform = transforms[0]
    for transform in transforms[1:]:
        final_transform = torch.matmul(transform, final_transform)
    return final_transform


def get_transform_from_rotation_translation(rotation, translation):
    r"""
    [PyTorch/Numpy] Get rigid transform matrix from rotation matrix and translation vector.

    :param rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :param translation: torch.Tensor (3,) or numpy.ndarray (3,)
    :return transform: torch.Tensor (4, 4) or numpy.ndarray (4, 4)
    """
    if isinstance(rotation, torch.Tensor):
        transform = torch.eye(4).to(rotation.device)
    else:
        transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform):
    r"""
    [PyTorch/Numpy] Get rotation matrix and translation vector from rigid transform matrix.

    :param transform: torch.Tensor (4, 4) or numpy.ndarray (4, 4)
    :return rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :return translation: torch.Tensor (3,) or numpy.ndarray (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def random_sample_rotation(rotation_factor=1):
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def random_sample_transform(points, rotation_range=360):
    r"""
    R: generate a random orthogonal matrix (R^-1 = R^T)
    """
    transform = np.eye(4)
    axis = np.random.rand(3) - 0.5  # (-0.5, 0.5)
    theta = rotation_range * np.pi / 180.0 * (np.random.rand(1) - 0.5)  # (-rotation_range / 2, rotation_range / 2)
    rotation = expm(np.cross(np.eye(3), axis / norm(axis) * theta))
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation.dot(-np.mean(points, axis=0))
    return transform


# Sampling methods

def random_sample_keypoints(points, feats, num_keypoint):
    num_point = points.shape[0]
    if num_point > num_keypoint:
        indices = np.random.choice(num_point, num_keypoint, replace=False)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_scores(points, feats, scores, num_keypoint):
    num_point = points.shape[0]
    if num_point > num_keypoint:
        indices = np.argsort(-scores)[:num_keypoint]
        points = points[indices]
        feats = feats[indices]
    return points, feats


def random_sample_keypoints_with_scores(points, feats, scores, num_keypoint):
    num_point = points.shape[0]
    if num_point > num_keypoint:
        indices = np.arange(num_point)
        probs = scores / np.sum(scores)
        indices = np.random.choice(indices, num_keypoint, replace=False, p=probs)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_nms(points, feats, scores, num_keypoint, radius):
    num_point = points.shape[0]
    if num_point > num_keypoint:
        radius2 = radius ** 2
        masks = np.ones(num_point, dtype=np.bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_point):
            if masks[i]:
                indices.append(i)
                if len(indices) == num_keypoint:
                    break
                if i + 1 < num_point:
                    current_masks = np.sum((sorted_points[i+1:] - sorted_points[i]) ** 2, axis=1) < radius2
                    masks[i+1:] = masks[i+1:] & ~current_masks
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats


def random_sample_keypoints_with_nms(points, feats, scores, num_keypoint, radius):
    num_point = points.shape[0]
    if num_point > num_keypoint:
        radius2 = radius ** 2
        masks = np.ones(num_point, dtype=np.bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_point):
            if masks[i]:
                indices.append(i)
                if i + 1 < num_point:
                    current_masks = np.sum((sorted_points[i+1:] - sorted_points[i]) ** 2, axis=1) < radius2
                    masks[i+1:] = masks[i+1:] & ~current_masks
        indices = np.array(indices)
        if len(indices) > num_keypoint:
            sorted_scores = scores[sorted_indices]
            scores = sorted_scores[indices]
            probs = scores / np.sum(scores)
            indices = np.random.choice(indices, num_keypoint, replace=False, p=probs)
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats

def unique_with_inds(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

@torch.no_grad()
def find_knn(gpu_index, locs, neighbor=32):
    n_points = locs.shape[0]
    # Search with torch GPU using pre-allocated arrays
    new_d_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.float32)
    new_i_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.int64)

    gpu_index.add(locs)

    gpu_index.search(locs, neighbor, new_d_torch_gpu, new_i_torch_gpu)
    gpu_index.reset()
    new_d_torch_gpu = torch.sqrt(new_d_torch_gpu)

    return new_d_torch_gpu, new_i_torch_gpu

# NOTE fastest way to cal geodesic distance
@torch.no_grad()
def cal_geodesic_vectorize(
   query_points, locs_float_, max_step=128, neighbor=64, radius=0.05
):

    locs_float_b = locs_float_
    n_queries=query_points.shape[0]
    n_points = locs_float_b.shape[0]

    #distances_arr, indices_arr = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

    neighbor = min(neighbor, locs_float_b.shape[0]-1)

    distances_arr,indices_arr = get_knn_indices(locs_float_b,locs_float_b, neighbor,return_distance=True)  # (num_proposal, max_point)
    # NOTE nearest neigbor is themself -> remove first element
    distances_arr = distances_arr[:, 1:]
    indices_arr = indices_arr[:, 1:]

    geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_.device) - 1
    visited = torch.zeros((n_queries, n_points), dtype=torch.int32, device=locs_float_.device)

    distances, indices = get_knn_indices(locs_float_b,query_points, neighbor,return_distance=True)  # (num_proposal, max_point)

    cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

    queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
    points_inds = indices[queries_inds, neighbors_inds]  # n_temp
    points_distances = distances[queries_inds, neighbors_inds]  # n_temp

    for step in range(max_step):
        # NOTE find unique indices for each query
        stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)
        _, unique_inds = unique_with_inds(stack_pointquery_inds)

        points_inds = points_inds[unique_inds] 
        queries_inds = queries_inds[unique_inds]
        points_distances = points_distances[unique_inds]

        # NOTE update geodesic and visited look-up table
        geo_dist[queries_inds, points_inds] = points_distances
        visited[queries_inds, points_inds] = 1

        # NOTE get new neighbors
        distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
        distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

        # NOTE trick to repeat queries indices for new neighbor
        queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

        # NOTE condition: no visited and radius and indices
        visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
        cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond == 0)  # n_temp x n_neighbors

        # NOTE filter
        temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2

        if len(temp_inds) == 0:  # no new points:
            break

        points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
        points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
        queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2

    return geo_dist
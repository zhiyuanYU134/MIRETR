import time,random
import numpy as np
import torch
import torch.nn as nn

from vision3d.utils.point_cloud_utils import pairwise_distance, apply_transform,get_knn_indices,cal_geodesic_vectorize
from vision3d.utils.torch_utils import index_select
from vision3d.modules.attention.rpe_attention import InstanceAwareTransformer
from vision3d.modules.attention.positional_embedding import SinusoidalPositionalEmbedding
from vision3d.modules.registration.modules import WeightedProcrustes
from vision3d.utils.registration_utils import compute_add_error



def random_triplet(high, size):
    triplets = torch.randint(low=0, high=high, size=(int(size), 3))
    local_dup_check = (triplets - triplets.roll(1, 1) != 0).all(dim=1)
    triplets = triplets[local_dup_check]
    return triplets



class InstanceAwareGeoTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_head, blocks, bin_size_d, bin_size_a, angle_k,max_neighboor,geodesic_radis):
        super(InstanceAwareGeoTransformer, self).__init__()
        self.bin_size_d = bin_size_d
        self.bin_size_a = bin_size_a
        self.bin_factor_a = 180. / (self.bin_size_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.rde_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rae_proj = nn.Linear(hidden_dim, hidden_dim)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = InstanceAwareTransformer(blocks, hidden_dim, num_head, dropout=0.1, activation_fn='relu')
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.rdistance_invarint_proj = nn.Linear(hidden_dim, hidden_dim)
        self.max_neighboors=max_neighboor
        self.bin_size_geo=bin_size_d/2
        self.geodesic_radis=geodesic_radis



    def _get_geometric_geodesic_embeddings(self, points, Radis,geodesic=True):
        with torch.no_grad():
            node_knn_distance,node_knn_indices = get_knn_indices(points, points, self.max_neighboors,return_distance=True)  # (N, num_neighbors)
            node_knn_masks=torch.gt(node_knn_distance, Radis*(0.5))# (N, num_neighbors)
            node_knn_points=points[node_knn_indices]# (N, num_neighbors,3)            
            rde_indices = node_knn_distance / self.bin_size_d
            knn_points =node_knn_points[:,1:4,:]# (N, k, 3)
            ref_vectors = knn_points - points.unsqueeze(1)  # (N, k, 3)
            anc_vectors = node_knn_points.unsqueeze(1) - points.unsqueeze(1).unsqueeze(2)  # (N,1, num_neighbors,3)
            ref_vectors = ref_vectors[node_knn_indices].unsqueeze(1)  # (N,1, num_neighbors, k, 3)
            anc_vectors = anc_vectors.unsqueeze(3).expand(anc_vectors.shape[0], anc_vectors.shape[1], anc_vectors.shape[2], self.angle_k, 3)# (N,1, num_neighbors,k,3)
            sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (N,1, num_neighbors,k)
            cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (N,1, num_neighbors,k)
            angles = torch.atan2(sin_values, cos_values)  # (N,1, num_neighbors,k)
            rae_indices = angles * self.bin_factor_a

        rde = self.embedding(rde_indices.unsqueeze(1) )  # (N,1, num_neighbors, C)
        rde = self.rde_proj(rde)  # (N,1, num_neighbors, C)

        rae = self.embedding(rae_indices)  # (N,1, num_neighbors,k, C)
        rae = self.rae_proj(rae)  # (N,1, num_neighbors,k, C)
        rae = rae.max(dim=3)[0]  # (N,1, num_neighbors, C)

        rge = rde + rae # (N,1, num_neighbors, C)

        if geodesic:
            with torch.no_grad():
                neighboors=min(self.max_neighboors, points.shape[0])
                geo_dists= torch.zeros((len(points), neighboors), dtype=torch.float32, device=points.device)
                geo_dist = cal_geodesic_vectorize(points,points,max_step=32, neighbor=neighboors,radius=self.geodesic_radis)  # (N, num)
                max_geo_dist_context = torch.max(geo_dist, dim=1)[0]  # (N)
                max_geo_val = torch.max(max_geo_dist_context)
                max_geo_dist_context[max_geo_dist_context < 0] = max_geo_val  # NOTE assign very big value to invalid queries
                max_geo_dist_context = max_geo_dist_context[:, None].expand(geo_dist.shape[0], geo_dist.shape[1])  # (N, num)
                cond = geo_dist < 0
                geo_dist[cond] = max_geo_dist_context[cond]# (N, num)
                for i in range(len(points)):
                    geo_dists[i]=geo_dist[i][node_knn_indices[i]] 
                geo_dist=geo_dists
                geo_dists=geo_dists/ self.bin_size_geo
            rdistance_invarint_e = self.embedding(geo_dists.unsqueeze(1) )  # (N,1, num_neighbors, C)
            rdistance_invarint_e = self.rdistance_invarint_proj(rdistance_invarint_e)  # (N,1, num_neighbors, C)
            rdistance_invarint_e+=rde
            return rge,rdistance_invarint_e,node_knn_indices,node_knn_masks#(N,1, num_neighbors, C) (N,1, num_neighbors, C), (N, num_neighbors)， (N, num_neighbors)

        return rge,node_knn_indices,node_knn_masks# (N,1, num_neighbors, C), (N, num_neighbors)， (N, num_neighbors)

    def forward(self, ref_points, src_points, ref_feats, src_feats, Radis):
        r"""
        Coarse Transformer with Relative Distance Embedding.

        :param ref_points: torch.Tensor (N, 3)
        :param src_points: torch.Tensor (M, 3)
        :param ref_feats: torch.Tensor (N, C)
        :param src_feats: torch.Tensor (M, C)
        :param ref_masks: torch.BoolTensor (N) (default: None)
        :param src_masks: torch.BoolTensor (M) (default: None)
        :return ref_feats: torch.Tensor (N, C)
        :return src_feats: torch.Tensor (M, C)
        :return ref_node_knn_indices: torch.Tensor (N, num_neighbors)
        :return src_node_knn_indices: torch.Tensor (M, num_neighbors)
        :return pred_masks_list: 
        """
        
        ref_embeddings,cross_postion_embedding,ref_node_knn_indices ,ref_node_knn_masks= self._get_geometric_geodesic_embeddings(ref_points, Radis,True)
        src_embeddings,src_node_knn_indices ,src_node_knn_masks = self._get_geometric_geodesic_embeddings(src_points, Radis,False)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        ref_feats, src_feats,pred_masks_list = self.transformer(
                ref_feats, src_feats, ref_embeddings, src_embeddings, cross_postion_embedding,ref_node_knn_indices,src_node_knn_indices,masks0=None, masks1=None
            )
        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)
        return ref_feats, src_feats,ref_node_knn_indices,src_node_knn_indices,pred_masks_list



class CoarseTargetGenerator(nn.Module):
    def __init__(self, num_corr, overlap_thresh=0.1):
        super(CoarseTargetGenerator, self).__init__()
        self.num_corr = num_corr
        self.overlap_thresh = overlap_thresh

    def forward(self, gt_corr_indices, gt_corr_overlaps):
        gt_corr_masks = torch.gt(gt_corr_overlaps, self.overlap_thresh)
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]
        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]
        if gt_corr_indices.shape[0] > self.num_corr:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices, self.num_corr, replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            gt_ref_corr_indices = gt_ref_corr_indices[sel_indices.long()]
            gt_src_corr_indices = gt_src_corr_indices[sel_indices.long()]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices.long()]


        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps
class CoarseMatching(nn.Module):
    def __init__(
            self,
            num_proposal,
            dual_softmax=True
    ):
        super(CoarseMatching, self).__init__()
        self.num_proposal = num_proposal
        self.dual_softmax = dual_softmax

    def forward(self, ref_feats, src_feats):
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=self.num_proposal, largest=True)
        ref_corr_indices = corr_indices // matching_scores.shape[1]
        src_corr_indices = corr_indices % matching_scores.shape[1]

        return ref_corr_indices, src_corr_indices, corr_scores




class InstanceAwarePointMatching(nn.Module):
    def __init__(
            self,
            cluster_thre,
            cluster_refine,
            max_num_corr,
            k,
            mutual=False,
            with_slack=False,
            threshold=0.,
            conditional=False,
            matching_radius=0.1,
            min_num_corr=3,
            num_registration_iter=5
    ):
        super(InstanceAwarePointMatching, self).__init__()
        self.max_num_corr = max_num_corr
        self.k = k
        self.mutual = mutual
        self.with_slack = with_slack
        self.threshold = threshold
        self.conditional = conditional
        self.matching_radius = matching_radius
        self.procrustes = WeightedProcrustes(return_transform=True)
        self.min_num_corr = min_num_corr
        self.num_registration_iter = num_registration_iter
        self.cluster_thre=cluster_thre
        self.cluster_refine=cluster_refine

    def compute_score_map_and_corr_map(
            self,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map,
            node_corr_scores
    ):
        matching_score_map = torch.exp(matching_score_map)
        corr_mask_map = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        num_proposal, ref_length, src_length = matching_score_map.shape
        proposal_indices = torch.arange(num_proposal).cuda()

        ref_topk_scores, ref_topk_indices = matching_score_map.topk(k=self.k, dim=2)  # (B, N, K)
        ref_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, ref_length, self.k)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(num_proposal, ref_length, self.k)
        ref_score_map = torch.zeros_like(matching_score_map)
        ref_score_map[ref_proposal_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        if self.with_slack:
            ref_score_map = ref_score_map[:, :-1, :-1]
        ref_corr_map = torch.logical_and(torch.gt(ref_score_map, self.threshold), corr_mask_map)

        src_topk_scores, src_topk_indices = matching_score_map.topk(k=self.k, dim=1)  # (B, K, N)
        src_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, self.k, src_length)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(num_proposal, self.k, src_length)
        src_score_map = torch.zeros_like(matching_score_map)
        src_score_map[src_proposal_indices, src_topk_indices, src_indices] = src_topk_scores
        if self.with_slack:
            src_score_map = src_score_map[:, :-1, :-1]
        src_corr_map = torch.logical_and(torch.gt(src_score_map, self.threshold), corr_mask_map)
        score_map = (ref_score_map + src_score_map) / 2

        if self.mutual:
            corr_map = torch.logical_and(ref_corr_map, src_corr_map)
        else:
            corr_map = torch.logical_or(ref_corr_map, src_corr_map)

        if self.conditional:
            node_corr_scores = node_corr_scores.view(-1, 1, 1)
            score_map = score_map * node_corr_scores
        
        return score_map, corr_map

    def cluster_transform(self,estimated_transforms,src_corr_points, ref_corr_points, corr_scores,corr_vote,src_points_m,thresold):
        #select candidates which are isolated        
        Rt_scores=compute_add_error(estimated_transforms,estimated_transforms,src_points_m)
        Rt_score_mask=torch.lt(Rt_scores,thresold)
        index=torch.nonzero(Rt_score_mask).reshape(-1)
        index=torch.unique(index)
        est_trans_mask=torch.zeros(len(estimated_transforms))
        est_trans_mask[index]=1
        est_trans_mask=est_trans_mask==1
        est_trans_mask_inv=torch.logical_not(est_trans_mask)
        estimated_transforms_inv=estimated_transforms[est_trans_mask_inv]

        
        #candidate selection  
        estimated_transforms=estimated_transforms[est_trans_mask]
        src_corr_points=src_corr_points[est_trans_mask]
        ref_corr_points=ref_corr_points[est_trans_mask]
        corr_scores=corr_scores[est_trans_mask]
        corr_vote=corr_vote[est_trans_mask]
        Rt_scores=compute_add_error(estimated_transforms,estimated_transforms,src_points_m)
        all_ref_corr_points = torch.zeros((1,3)).cuda()
        all_src_corr_points = torch.zeros((1,3)).cuda()
        all_corr_scores = torch.zeros((1)).cuda()

        estimated_transforms_good=[]
        len_corr=[]
        while len(estimated_transforms)>0:
            Rt_score=Rt_scores[0]
            Rt_score_mask=torch.lt(Rt_score,thresold)
            Rt_score_mask_inv=torch.logical_not(Rt_score_mask)
            src_corr_points_tmp=src_corr_points[Rt_score_mask].reshape(-1,3)
            ref_corr_points_tmp=ref_corr_points[Rt_score_mask].reshape(-1,3)
            len_corr.append(len(ref_corr_points_tmp))
            corr_scores_tmp=corr_scores[Rt_score_mask].reshape(-1)
            all_ref_corr_points=torch.cat((all_ref_corr_points,ref_corr_points_tmp),dim=0)
            all_src_corr_points=torch.cat((all_src_corr_points,src_corr_points_tmp),dim=0)
            all_corr_scores=torch.cat((all_corr_scores,corr_scores_tmp),dim=0)
            estimated_transform =estimated_transforms[0] 
            estimated_transforms_good.append(estimated_transform)
            Rt_scores=Rt_scores[Rt_score_mask_inv]
            Rt_scores=Rt_scores[:,Rt_score_mask_inv]
            estimated_transforms=estimated_transforms[Rt_score_mask_inv]
            src_corr_points=src_corr_points[Rt_score_mask_inv]
            ref_corr_points=ref_corr_points[Rt_score_mask_inv]
            corr_scores=corr_scores[Rt_score_mask_inv]
            if len(estimated_transforms)==1:
                
                src_corr_points=src_corr_points.reshape(-1,3)
                ref_corr_points=ref_corr_points.reshape(-1,3)
                corr_scores=corr_scores.reshape(-1)
                len_corr.append(len(ref_corr_points))
                all_ref_corr_points=torch.cat((all_ref_corr_points,ref_corr_points),dim=0)
                all_src_corr_points=torch.cat((all_src_corr_points,src_corr_points),dim=0)
                all_corr_scores=torch.cat((all_corr_scores,corr_scores),dim=0)
                estimated_transform =estimated_transforms[0]
                estimated_transforms_good.append(estimated_transform)
                break
        #refine candidate
        if len(estimated_transforms_good)>1:
            estimated_transforms=torch.stack(estimated_transforms_good)
            max_corr=np.max(np.array(len_corr))
            all_ref_corr_points=all_ref_corr_points[1:,:]
            all_src_corr_points=all_src_corr_points[1:,:]
            all_corr_scores=all_corr_scores[1:]
            local_ref_corr_points = torch.zeros(len(len_corr)*max_corr, 3).cuda()
            local_src_corr_points = torch.zeros(len(len_corr)*max_corr, 3).cuda()
            local_corr_scores = torch.zeros(len(len_corr)*max_corr).cuda()
            target_chunks = [(i * max_corr, i * max_corr + y ) for i, (y) in enumerate(len_corr)]
            indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
            indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total, 3)
            indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (total, 3)
            local_ref_corr_points.index_put_([indices0, indices1], all_ref_corr_points)
            local_ref_corr_points = local_ref_corr_points.view(len(len_corr), max_corr, 3)
            local_src_corr_points.index_put_([indices0, indices1], all_src_corr_points)
            local_src_corr_points = local_src_corr_points.view(len(len_corr), max_corr, 3)
            local_corr_scores.index_put_([indices], all_corr_scores)
            local_corr_scores = local_corr_scores.view(len(len_corr), max_corr)
            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)
            for _ in range(4):
                aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)
                corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
                inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2)
                local_corr_scores = local_corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)
            estimated_transforms=torch.cat((estimated_transforms,estimated_transforms_inv),dim=0)
        elif len(estimated_transforms_good)==1:
            estimated_transforms=estimated_transforms_good[0].unsqueeze(0)
            estimated_transforms=torch.cat((estimated_transforms,estimated_transforms_inv),dim=0)
        else:
            estimated_transforms=estimated_transforms_inv#torch.eye(4).unsqueeze(0).cuda()
        
        return estimated_transforms
    
    def Cal_Inliers(self,estimated_transforms,ref_points_m,src_points_m):
        all_aligned_src_points = apply_transform(src_points_m.unsqueeze(0), estimated_transforms)
        if len(ref_points_m)>self.max_num_corr:
            inds = torch.LongTensor(random.sample(range(len(ref_points_m)), self.max_num_corr)).cuda()
            ref_points_m=ref_points_m[inds]
        inliers=torch.zeros(len(estimated_transforms)).cuda()
        max_instance=8
        head=0
        for i in range((len(estimated_transforms)//max_instance)+1):
            if head+max_instance>len(estimated_transforms):
                end=len(estimated_transforms)
            else:
                end=head+max_instance
            aligned_src_points=all_aligned_src_points[head:end]
            src_closest_distance,src_closest_indices = get_knn_indices(ref_points_m,aligned_src_points.reshape(-1,3),  1,return_distance=True) 
            inlier_masks=torch.lt(src_closest_distance.reshape(-1), self.matching_radius)
            inlier_masks=inlier_masks.reshape(end-head,len(src_points_m))
            inliers[head:end]=inlier_masks.sum(dim=1)
            head+=max_instance
        return inliers

    def fast_compute_all_transforms(
            self,
            ref_knn_points,
            src_knn_points,
            score_map,
            corr_map,
            ref_points_m,
            src_points_m
    ):
        proposal_indices, ref_indices, src_indices = torch.nonzero(corr_map, as_tuple=True)
        all_ref_corr_points = ref_knn_points[proposal_indices, ref_indices]
        all_src_corr_points = src_knn_points[proposal_indices, src_indices]
        all_corr_scores = score_map[proposal_indices, ref_indices, src_indices]

        # torch.nonzero is row-major, so the correspondences from the same proposal are consecutive.
        # find the first occurrence of each proposal index, then the chunk of this proposal can be obtained.
        unique_masks = torch.ne(proposal_indices[1:], proposal_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [proposal_indices.shape[0]]

        chunks = [(x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.min_num_corr]
        num_proposal = len(chunks)
        if num_proposal > 0:
            indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
            stacked_ref_corr_points = index_select(all_ref_corr_points, indices, dim=0)  # (total, 3)
            stacked_src_corr_points = index_select(all_src_corr_points, indices, dim=0)  # (total, 3)
            stacked_corr_scores = index_select(all_corr_scores, indices, dim=0)  # (total,)

            max_corr = np.max([y - x for x, y in chunks])
            target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
            indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
            indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total, 3)
            indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (total, 3)

            local_ref_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_ref_corr_points.index_put_([indices0, indices1], stacked_ref_corr_points)
            local_ref_corr_points = local_ref_corr_points.view(num_proposal, max_corr, 3)
            local_src_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_src_corr_points.index_put_([indices0, indices1], stacked_src_corr_points)
            local_src_corr_points = local_src_corr_points.view(num_proposal, max_corr, 3)
            local_corr_scores = torch.zeros(num_proposal * max_corr).cuda()
            local_corr_scores.index_put_([indices], stacked_corr_scores)
            local_corr_scores = local_corr_scores.view(num_proposal, max_corr)

            #calculate candidate's pose
            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)
            aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)
            corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
            inlier_masks = torch.lt(corr_distances, (self.matching_radius)  ** 2).float() # (P, N)
            
            #remove bad poses
            inlier_index_masks= torch.gt(inlier_masks.sum(dim=1),self.min_num_corr)
            local_src_corr_points=local_src_corr_points[inlier_index_masks]
            local_ref_corr_points=local_ref_corr_points[inlier_index_masks]
            local_corr_scores=local_corr_scores[inlier_index_masks]
            inlier_masks=inlier_masks[inlier_index_masks]
            estimated_transforms=estimated_transforms[inlier_index_masks]
            
            #refine candidate's pose
            cur_corr_scores = local_corr_scores * inlier_masks.float()
            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, cur_corr_scores)
            for _ in range(self.num_registration_iter - 1):
                aligned_src_corr_points = apply_transform(local_src_corr_points, estimated_transforms)
                corr_distances = torch.sum((local_ref_corr_points - aligned_src_corr_points) ** 2, dim=2)
                inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2)
                cur_corr_scores = local_corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, cur_corr_scores)

            #remove bad poses
            inlier_index_masks= torch.gt(inlier_masks.sum(dim=1),self.min_num_corr)
            local_src_corr_points=local_src_corr_points[inlier_index_masks]
            local_ref_corr_points=local_ref_corr_points[inlier_index_masks]
            local_corr_scores=local_corr_scores[inlier_index_masks]
            cur_corr_scores=cur_corr_scores[inlier_index_masks]
            estimated_transforms=estimated_transforms[inlier_index_masks]

            #remove bad correspondences
            num_proposal, max_corr, _=local_ref_corr_points.shape
            all_ref_corr_points = local_ref_corr_points.reshape(num_proposal*max_corr,3)
            all_src_corr_points = local_src_corr_points.reshape(num_proposal*max_corr,3)
            all_corr_scores = local_corr_scores.reshape(num_proposal*max_corr)
            all_corr_masks=torch.gt(all_corr_scores,0)
            all_ref_corr_points=all_ref_corr_points[all_corr_masks]
            all_src_corr_points=all_src_corr_points[all_corr_masks]
            all_corr_scores=all_corr_scores[all_corr_masks]

            #sort poses by inliers
            aligned_src_corr_points = apply_transform(all_src_corr_points.unsqueeze(0), estimated_transforms)
            corr_distances = torch.sum((all_ref_corr_points.unsqueeze(0) - aligned_src_corr_points) ** 2, dim=2)
            inlier_masks = torch.lt(corr_distances, (self.matching_radius) ** 2).float()
            corr_vote=inlier_masks.sum(dim=1)
            sorted_inlier,inlier_indices=torch.sort(corr_vote,descending=True)
            estimated_transforms=estimated_transforms[inlier_indices]
            corr_vote=corr_vote[inlier_indices]
            local_ref_corr_points = local_ref_corr_points[inlier_indices]
            local_src_corr_points = local_src_corr_points[inlier_indices]
            cur_corr_scores = cur_corr_scores[inlier_indices]

            estimated_transforms=self.cluster_transform(estimated_transforms,local_src_corr_points, local_ref_corr_points, cur_corr_scores,corr_vote,src_points_m,self.cluster_thre)
            #remove bad poses
            if self.cluster_refine:
                if len(estimated_transforms)>0:
                    inliers=self.Cal_Inliers(estimated_transforms,ref_points_m,src_points_m)
                    max_inliers=torch.max(inliers)
                    estimated_transforms_score_mask=inliers>max_inliers*0.1
                    estimated_transforms=estimated_transforms[estimated_transforms_score_mask]
            return estimated_transforms,all_ref_corr_points,all_src_corr_points,all_corr_scores
        else:
            estimated_transforms = self.procrustes(all_src_corr_points, all_ref_corr_points, all_corr_scores)

            aligned_src_corr_points = apply_transform(all_src_corr_points, estimated_transforms)
            corr_distances = torch.sum((all_ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
            inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2).float()
            cur_corr_scores = all_corr_scores * inlier_masks
            estimated_transforms = self.procrustes(all_src_corr_points, all_ref_corr_points, cur_corr_scores)
            for _ in range(self.num_registration_iter - 1):
                aligned_src_corr_points = apply_transform(all_src_corr_points, estimated_transforms)
                corr_distances = torch.sum((all_ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
                inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2)
                cur_corr_scores = all_corr_scores * inlier_masks.float()
                estimated_transforms = self.procrustes(all_src_corr_points, all_ref_corr_points, cur_corr_scores)
            return estimated_transforms.unsqueeze(0),all_ref_corr_points,all_src_corr_points,all_corr_scores

    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map,
            node_corr_scores,
            ref_points_m,
            src_points_m
            
    ):
        """
        :param ref_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param src_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param ref_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param src_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param matching_score_map: torch.Tensor (num_proposal, num_point, num_point)
        :param node_corr_scores: torch.Tensor (num_proposal)

        :return estimated_transforms: torch.LongTensor (N,4,4)
        :return ref_corr_indices: torch.LongTensor (self.num_corr,)
        :return src_corr_indices: torch.LongTensor (self.num_corr,)
        :return corr_scores: torch.Tensor (self.num_corr,)
        """
        score_map, corr_map = self.compute_score_map_and_corr_map(
            ref_knn_masks, src_knn_masks, matching_score_map, node_corr_scores
        )
        estimated_transforms,all_ref_corr_points,all_src_corr_points,all_corr_scores= self.fast_compute_all_transforms(
                ref_knn_points, src_knn_points, score_map, corr_map,ref_points_m,src_points_m
            )
        return estimated_transforms,all_ref_corr_points,all_src_corr_points,all_corr_scores


class LearnableLogOptimalTransport(nn.Module):
    def __init__(self, num_iter, inf=1e12):
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iter = num_iter
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iter):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks, col_masks):
        r"""
        Optimal transport with Sinkhorn.

        :param scores: torch.Tensor (B, M, N)
        :param row_masks: torch.Tensor (B, M)
        :param col_masks: torch.Tensor (B, N)
        :return matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape
        ninf = torch.tensor(-self.inf).cuda()

        padded_row_masks = torch.zeros(batch_size, num_row + 1, dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(batch_size, num_col + 1, dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)

        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))
        padded_scores[padded_score_masks] = ninf

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(batch_size, num_row + 1).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = ninf

        log_nu = torch.empty(batch_size, num_col + 1).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = ninf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iter={})'.format(self.num_iter)
        return format_string



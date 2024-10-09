from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import copy
import numpy as np
from vision3d.utils.point_cloud_utils import apply_transform, pairwise_distance,rotation_to_axis_angle
from vision3d.utils.registration_utils import extract_corr_indices_from_scores, compute_registration_error,to_o3d_pcd
from vision3d.modules.loss.circle_loss import WeightedCircleLoss


class CoarseMatchingLoss(nn.Module):
    def __init__(self, config):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(config.coarse_circle_loss_positive_margin,
                                                       config.coarse_circle_loss_negative_margin,
                                                       config.coarse_circle_loss_positive_optimal,
                                                       config.coarse_circle_loss_negative_optimal,
                                                       config.coarse_circle_loss_log_scale)
        self.pos_thresh = config.coarse_circle_loss_positive_threshold
        self.num_proposal = config.coarse_matching_num_proposal

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.pos_thresh)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, config):
        super(FineMatchingLoss, self).__init__()
        self.pos_radius = config.fine_sinkhorn_loss_positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points_ori = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transforms = data_dict['transform']

        transform=transforms[0]
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points_ori, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.pos_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)

        for i in range(len(transforms)-1):
            transform=transforms[i+1]
            src_node_corr_knn_points = apply_transform(src_node_corr_knn_points_ori, transform)
            dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
            gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
            gt_corr_map_tmp = torch.lt(dists, self.pos_radius ** 2)
            gt_corr_map_tmp= torch.logical_and(gt_corr_map_tmp, gt_masks)
            gt_corr_map=torch.logical_or(gt_corr_map_tmp, gt_corr_map)
            
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss

class InstanceMaskLoss(nn.Module):
    def __init__(self, config):
        super(InstanceMaskLoss, self).__init__()

    
    def dice_loss(self,inputs,targets):

        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)  
        return loss.mean()

    def forward(self, output_dict):
        pred_masks_list=output_dict['pred_masks_list']
        gt_masks=output_dict['gt_masks']
        ref_node_corr_indices=output_dict['ref_node_corr_indices']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        mask_bce_loss = torch.tensor([0.0], device=gt_masks.device)
        mask_dice_loss = torch.tensor([0.0], device=gt_masks.device)
        for pred_masks in pred_masks_list:
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_masks[gt_ref_node_corr_indices], gt_masks[gt_ref_node_corr_indices].float())
            mask_dice_loss += self.dice_loss(pred_masks[gt_ref_node_corr_indices],gt_masks[gt_ref_node_corr_indices].float())
        
        mask_bce_loss = mask_bce_loss / len(pred_masks_list)
        mask_dice_loss = mask_dice_loss / len(pred_masks_list)

     

        return mask_bce_loss,mask_dice_loss


class OverallLoss(nn.Module):
    def __init__(self, config):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(config)
        self.fine_loss = FineMatchingLoss(config)
        """ 
        self.dhvr_loss = DHVRLoss(config)
        self.RT_loss = RT_loss(config) """
        self.mask_loss=InstanceMaskLoss(config)
        self.weight_coarse_loss = config.weight_coarse_loss
        self.weight_fine_loss = config.weight_fine_loss
        self.weight_mask_loss = config.weight_mask_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)
        mask_bce_loss,mask_dice_loss=self.mask_loss(output_dict)
        
        loss = self.weight_coarse_loss * coarse_loss +self.weight_fine_loss *fine_loss+self.weight_mask_loss*(mask_bce_loss+mask_dice_loss)
        result_dict = {
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'mask_bce_loss': mask_bce_loss,
            'mask_dice_loss':mask_dice_loss,
            'loss': loss
        }

        return result_dict


def  iou(box1,box2):
	'''
	3D IoU计算
	box表示形式：[x1,y1,z1,x2,y2,z2] 分别是两对角点的坐标
	'''
	in_w = min(box1[3],box2[3]) - max(box1[0],box2[0])
	in_l = min(box1[4],box2[4]) - max(box1[1],box2[1])
	in_h = min(box1[5],box2[5]) - max(box1[2],box2[2])

	inter = 0 if in_w < 0 or in_l < 0 or in_h < 0 else in_w * in_l * in_h
	union = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2]) + (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])  - inter
	iou = inter / union
	return iou




class Evaluator(nn.Module):
    def __init__(self, config):
        super(Evaluator, self).__init__()
        self.positive_overlap = config.coarse_matching_positive_overlap
        self.positive_radius = config.fine_matching_positive_radius
        self.re_thre=config.re_thre
        self.te_thre=config.te_thre
        self.add_s=0.1
        self.iou_ratio=0.7

    @torch.no_grad()
    def evaluate_coarse(self, output_dict, data_dict):
        ref_length_c = len(output_dict['ref_nodes'])
        src_length_c =  len(output_dict['src_nodes'])
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.positive_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.positive_radius).float().mean()
        return precision
    
    


    @torch.no_grad()
    def evaluate_registrations(self, est_transforms, transforms):
        recall_trans = torch.zeros(len(transforms))
        precision_pred = torch.zeros(len(est_transforms))
        recall_trans_index = torch.zeros(len(transforms))-1
        precision_pred_index = torch.zeros(len(est_transforms))-1
        recall_best=torch.zeros((len(transforms),2))+361
        if len(est_transforms)>0:
            for i in range( len(transforms)):
                transform=transforms[i]
                for j in range(len(est_transforms)):
                    rre, rte = compute_registration_error(transform, est_transforms[j])
                    if rre < self.re_thre and rte < self.te_thre:
                        precision_pred[j] = 1
                        recall_trans[i] = 1
                        precision_pred_index[j] = i
                        if rre<=recall_best[i][0] and rte<=recall_best[i][1]:
                            recall_trans_index[i] = j
                            recall_best[i][0]=rre
                            recall_best[i][1]=rte
            precision = precision_pred.sum() / len(precision_pred)
            recall = recall_trans.sum() / len(recall_trans)

            return precision, recall,recall_trans_index,precision_pred_index,recall_best
        else:
            return 0.0,0.0,recall_trans_index,precision_pred_index,recall_best
    
    @torch.no_grad()
    def evaluate_sym_registrations(self, est_transforms, transforms,src_points):
        recall_trans = torch.zeros(len(transforms))
        precision_pred = torch.zeros(len(est_transforms))
        recall_trans_index = torch.zeros(len(transforms))-1
        precision_pred_index = torch.zeros(len(est_transforms))-1
        recall_best=torch.zeros((len(transforms),1))+361
        cad=to_o3d_pcd(src_points)
        src_points=np.array(cad.points)
        ab = np.matmul(src_points, src_points.transpose(-1, -2))
        a2 = np.expand_dims(np.sum(src_points ** 2, axis=-1), axis=-1)
        b2 = np.expand_dims(np.sum(src_points ** 2, axis=-1), axis=-2)
        dist2 = a2 - 2 * ab + b2
        src_R=np.sqrt(dist2.max())
        if len(est_transforms)>0:
            for i in range( len(transforms)):
                transform=transforms[i]
                for j in range(len(est_transforms)):
                    
                    src_pcd_gt = copy.deepcopy(cad)
                    src_pcd_pred = copy.deepcopy(cad)
                    src_pcd_gt.transform(transform.cpu().numpy())
                    src_pcd_pred.transform(est_transforms[j].cpu().numpy())
                    aabb_gt = src_pcd_gt.get_axis_aligned_bounding_box() 
                    aabb_pred = src_pcd_pred.get_axis_aligned_bounding_box() 
                    box_points_gt_max=aabb_gt.get_max_bound()
                    box_points_gt_min=aabb_gt.get_min_bound()
                    box_points_gt=np.concatenate([box_points_gt_min,box_points_gt_max],axis=0)
                    box_points_pred_max=aabb_pred.get_max_bound()
                    box_points_pred_min=aabb_pred.get_min_bound()
                    box_points_pred=np.concatenate([box_points_pred_min,box_points_pred_max],axis=0)
                    point_iou=iou(box_points_gt,box_points_pred)
                    if point_iou>self.iou_ratio:
                        src_pcd_gt=np.array(src_pcd_gt.points)
                        src_pcd_pred=np.array(src_pcd_pred.points)
                        ab = np.matmul(src_pcd_gt, src_pcd_pred.transpose(-1, -2))
                        a2 = np.expand_dims(np.sum(src_pcd_gt ** 2, axis=-1), axis=-1)
                        b2 = np.expand_dims(np.sum(src_pcd_pred ** 2, axis=-1), axis=-2)
                        dist2 = a2 - 2 * ab + b2
                        dist2 =dist2.min(1)
                        dist2=np.sqrt(dist2)
                        avg = np.average(dist2)
                        if avg<self.add_s*src_R:
                            precision_pred[j] = 1
                            recall_trans[i] = 1
                            precision_pred_index[j] = i
                            if avg/src_R<=recall_best[i][0]:
                                recall_trans_index[i] = j
                                recall_best[i][0]=avg/src_R
            precision = precision_pred.sum() / len(precision_pred)
            recall = recall_trans.sum() / len(recall_trans)

            return precision, recall,recall_trans_index,precision_pred_index,recall_best
        else:
            return 0.0,0.0,recall_trans_index,precision_pred_index,recall_best

    def forward(self, output_dict, data_dict):

        est_transforms = output_dict['estimated_transforms']
        transforms = data_dict['transform']
        if data_dict['frag_id0']=="__SYM_NONE":
            precision, recall,recall_trans,precision_pred,recall_best = self.evaluate_registrations(est_transforms, transforms)
        else:
            precision, recall ,recall_trans,precision_pred,recall_best= self.evaluate_sym_registrations(est_transforms, transforms,output_dict['src_points_m'])
        if precision==0.0 and recall==0.0:
            F1_score=0.0
        else:
            F1_score=2 * (precision ) * (recall) / ((recall  +precision))


        result_dict = {
            'precision': precision,
            'recall': recall,
            'F1_score':F1_score,
            'recall_trans': recall_trans.long(),
            'recall_best':recall_best,
            'precision_pred': precision_pred.long(),

        }
        return result_dict

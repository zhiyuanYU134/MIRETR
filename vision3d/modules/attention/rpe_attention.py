# Attention layers for tensor shape (B, N, C).
# Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.torch_utils import get_dropout
from .vanilla_attention import AttentionOutput, TransformerLayer

class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_head != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_head` ({}).'.format(d_model, num_head))

        self.d_model = d_model
        self.num_head = num_head
        self.d_model_per_head = d_model // num_head

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = get_dropout(dropout)

    def _transpose_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_head, self.d_model_per_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def _transpose_rpe_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.num_head, self.d_model_per_head)
        x = x.permute(0, 3, 1, 2, 4)
        return x

    def forward(self, input_q, input_k, input_v, input_p, key_masks=None):
        r"""
        :param input_q: torch.Tensor (B, 1, C)
        :param input_k: torch.Tensor (B, M, C)
        :param input_v: torch.Tensor (B, M, C)
        :param input_p: torch.Tensor (B, 1, M, C), relative positional embedding
        :param key_masks: torch.Tensor (B, M), True if ignored, False if preserved
        :return hidden_states: torch.Tensor (B, C, N)
        """
        endpoints = {}

        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)
        p = self.proj_p(input_p)


        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)
        p = self._transpose_rpe_for_scores(p)  

        attention_scores_p = torch.matmul(q.unsqueeze(3), p.transpose(-1, -2)).squeeze(3)
        attention_scores_e = torch.matmul(q, k.transpose(-1, -2))
        
        attention_scores = attention_scores_e + attention_scores_p
        attention_scores = attention_scores / self.d_model_per_head ** 0.5
        if key_masks is not None:
            key_masks=key_masks.unsqueeze(1).unsqueeze(1)
            key_masks=key_masks.expand(key_masks.shape[0], self.num_head, key_masks.shape[2],key_masks.shape[3] )
            attention_scores = attention_scores.masked_fill(key_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        endpoints['attention_scores'] = attention_scores

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.d_model)

        return hidden_states, endpoints


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_head, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = get_dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states, memory_states, position_states, memory_masks=None):
        hidden_states, endpoints = self.attention(
            input_states, memory_states, memory_states, position_states, key_masks=memory_masks
        )
        hidden_states = self.linear(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, endpoints


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1, activation_fn='gelu', **kwargs):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_head, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn, **kwargs)

    def forward(self, input_states, memory_states, position_states, memory_masks=None):
        hidden_states, endpoints = self.attention(
            input_states, memory_states, position_states, memory_masks=memory_masks
        )
        output_states = self.output(hidden_states)
        return output_states, endpoints


class InstanceAwareTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_head, dropout=0.1, activation_fn='gelu'):
        super(InstanceAwareTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            elif block == 'cross':
                layers.append(TransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            elif block == 'mask':
                layers.append(RPETransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            else:
                raise ValueError('Unsupported block type "{}" in `RPEConditionalTransformer`.'.format(block))
        self.layers = nn.ModuleList(layers)
        self.mask_proj = nn.Sequential(
        nn.Linear(2*d_model, d_model),nn.LayerNorm(d_model), nn.ReLU() , 
        nn.Linear(d_model, 1))

        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model),nn.LayerNorm(d_model))

    def prediction_head(self, query, mask_feats,cross_position_embeddings):
        #(num_proposal,1 ,C) ,(num_proposal,1, num_neighbors, C)
        query=self.out_proj(query)
        all_ref_node_mask_features=torch.cat((mask_feats-query,cross_position_embeddings.squeeze(1)),dim=-1)
        pred_masks=self.mask_proj(all_ref_node_mask_features).squeeze(-1)
        pred_masks=pred_masks.sigmoid() 
        attn_mask = (pred_masks < 0.5).bool()
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        attn_mask = attn_mask.detach()
        return pred_masks, attn_mask

    def forward(self, feats0, feats1, embeddings0, embeddings1, cross_position_embeddings,ref_node_knn_indices,src_node_knn_indices,masks0=None, masks1=None):
        #(N, C),(M, C), (N,1, num_neighbors, C), (M,1, num_neighbors, C), (N,1, num_neighbors, C), (N, num_neighbors),(M, num_neighbors), (N, num_neighbors),(M, num_neighbors)
        pred_masks_list, attn_masks_list,mask_attention_score_list=[],[],[]
        for i, block in enumerate(self.blocks):
            if block == 'self':
                #(N, 1，C),(N,num_neighbors,  C), (N,1, num_neighbors, C),(N,1, num_neighbors, C)
                feats0, _ = self.layers[i](feats0.unsqueeze(1), feats0[ref_node_knn_indices], embeddings0, memory_masks=masks0)
                feats1, _ = self.layers[i](feats1.unsqueeze(1), feats1[src_node_knn_indices], embeddings1, memory_masks=masks1)
                feats0=feats0.squeeze(1)
                feats1=feats1.squeeze(1)
            elif block == 'cross' :
                feats0=feats0.unsqueeze(0)
                feats1=feats1.unsqueeze(0)
                feats0, _ = self.layers[i](feats0, feats1, memory_masks=None)#masks1
                feats1, _ = self.layers[i](feats1, feats0, memory_masks=None)#masks0
                feats0=feats0.squeeze(0)
                feats1=feats1.squeeze(0)
            else:
                #(N,1 ,C) ,(N, num_neighbors, C), (N,1, num_neighbors, C),(N, num_neighbors)
                ref_support_feature, endpoints = self.layers[i](feats0.unsqueeze(1), feats0[ref_node_knn_indices], cross_position_embeddings,memory_masks=masks0)#masks1
                mask_attention_score_list.append(endpoints['attention_scores'])
                ref_support_feature=ref_support_feature.squeeze(1)
                pred_masks, attn_masks=self.prediction_head(ref_support_feature.unsqueeze(1),ref_support_feature[ref_node_knn_indices],cross_position_embeddings)
                masks0=attn_masks
                pred_masks_list.append(pred_masks)
                attn_masks_list.append(attn_masks)
        return feats0, feats1,pred_masks_list



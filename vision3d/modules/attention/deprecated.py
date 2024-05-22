# Attention layers for tensor shape (B, C, N).
# Implemented with `nn.Conv1d` and `nn.InstanceNorm1d` (without affine).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.torch_utils import get_activation


def _create_dropout_layer(dropout):
    if dropout is not None:
        return nn.Dropout(dropout)
    else:
        return None


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_head != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_head` ({}).'.format(d_model, num_head))

        self.d_model = d_model
        self.num_head = num_head
        self.d_model_per_head = d_model // num_head

        self.proj_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        self.proj_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        self.proj_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)

        self.dropout = _create_dropout_layer(dropout)

    def _transpose_for_scores(self, x):
        x = x.view(x.shape[0], self.num_head, self.d_model_per_head, x.shape[-1])
        x = x.permute(0, 1, 3, 2)
        return x

    def forward(self, input_q, input_k, input_v, key_masks=None):
        '''
        :param input_q: torch.Tensor (B, C, N)
        :param input_k: torch.Tensor (B, C, M)
        :param input_v: torch.Tensor (B, C, M)
        :param key_masks: torch.Tensor (B, M), -inf if masked, 0 otherwise
        :return: hidden_states: torch.Tensor (B, C, N)
        '''
        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)

        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.d_model_per_head)
        if key_masks is not None:
            attention_scores = attention_scores + key_masks.unsqueeze(1)
        attention_scores = F.softmax(attention_scores, dim=-1)

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = hidden_states.permute(0, 1, 3, 2).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], self.d_model, hidden_states.shape[-1])

        return hidden_states


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_head, dropout=dropout)
        self.linear = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = _create_dropout_layer(dropout)
        self.norm = nn.InstanceNorm1d(d_model)

    def forward(self, input_states, memory_states, memory_masks=None):
        hidden_states = self.attention(input_states, memory_states, memory_states, attention_mask=memory_masks)
        hidden_states = self.linear(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states


class AttentionOutput(nn.Module):
    def __init__(self, d_model, dropout=0.1, activation_fn='gelu', **kwargs):
        super(AttentionOutput, self).__init__()
        self.expand = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.activation = get_activation(activation_fn, **kwargs)
        self.squeeze = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.dropout = _create_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states):
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        output_states = self.norm(input_states + hidden_states)
        return output_states


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1, activation_fn='gelu', **kwargs):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionLayer(d_model, num_head, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn, **kwargs)

    def forward(self, input_states, memory_states, memory_masks=None):
        hidden_states = self.attention(input_states, memory_states, memory_masks=memory_masks)
        output_states = self.output(hidden_states)
        return output_states


class AttentionalGNN(nn.Module):
    def __init__(self, blocks, d_model, num_head, dropout=0.1, activation_fn='gelu'):
        super(AttentionalGNN, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            if block == 'self' or block == 'cross':
                layers.append(TransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            else:
                raise ValueError('Unsupported block type "{}" in `AttentionalGNN`.'.format(block))
        self.layers = nn.ModuleList(layers)

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1 = self.layers[i](feats1, feats0, memory_masks=masks0)
        return feats0, feats1

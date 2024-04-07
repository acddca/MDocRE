# -*- coding:utf-8 -*-
# @Author: wangjl
# @Time: 2022/12/16 12:14
# @File: model_utils.py
# @Email: wangjl.nju.2020@gmail.com.
import copy
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linears = clone_module(nn.Linear(self.hidden_size, self.hidden_size), 3)

    def forward(self, inputs, attn_masks=None):
        bs, sl1, hs = inputs.shape
        assert hs == self.hidden_size

        q, k, v = [
            layer(x).view(bs, -1, self.num_heads, self.head_size).transpose(1, 2)
            for layer, x in zip(self.linears, (inputs, inputs, inputs))
        ]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        if attn_masks is not None:
            score_masks = (1 - attn_masks) * -1e30
            score_masks = score_masks.unsqueeze(dim=-1).unsqueeze(dim=-1)
            attn_scores = attn_scores + score_masks

        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.dropout(attn_probs)

        context_output = torch.matmul(attn_probs, v)
        context_output = context_output.permute(0, 2, 1, 3).contiguous()
        context_output = context_output.view(bs, -1, self.hidden_size)

        # (bs, seq_len, hs), (bs, num_heads, seq_len, seq_len)
        return context_output, attn_probs


class CrossMultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout_rate=0.5):
        super(CrossMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linears = clone_module(nn.Linear(self.hidden_size, self.hidden_size), 3)

    def forward(self, inputs1, inputs2):
        bs, sl1, hs = inputs1.shape
        _, sl2, hs = inputs2.shape
        assert hs == self.hidden_size
        assert sl1 == sl2

        q, k, v = [
            layer(x).view(bs, -1, self.num_heads, self.head_size).transpose(1, 2)
            for layer, x in zip(self.linears, (inputs1, inputs2, inputs2))
        ]

        # score_masks = (1 - attn_masks1) * -1e30
        # score_masks = score_masks.unsqueeze(dim=-1).unsqueeze(dim=-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # attn_scores = attn_scores + score_masks
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.dropout(attn_probs)

        context_output = torch.matmul(attn_probs, v)
        context_output = context_output.permute(0, 2, 1, 3).contiguous()
        context_output = context_output.view(bs, -1, self.hidden_size)

        return context_output, attn_probs


class OutputLayer(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(OutputLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, hidden_states, inputs):
        context_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.layer_norm(context_states + inputs)
        return hidden_states


class SelfAttentionLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.multi_attention_layer = MultiHeadAttention(self.hidden_size, self.num_heads, self.dropout_rate)
        self.output_layer = OutputLayer(self.hidden_size, self.dropout_rate)

    def forward(self, inputs, attn_masks):
        hidden_states, attentions = self.multi_attention_layer(inputs, attn_masks)
        hidden_states = self.output_layer(hidden_states, inputs)
        return hidden_states, attentions


class CrossAttentionLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout=0.5):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.multi_attention_layer = CrossMultiHeadAttention(self.hidden_size, self.num_heads, self.dropout_rate)
        self.output_layer = OutputLayer(self.hidden_size, self.dropout_rate)

    def forward(self, inputs1, inputs2):
        hidden_states, attentions = self.multi_attention_layer(inputs1, inputs2)
        hidden_states = self.output_layer(hidden_states, inputs2)
        return hidden_states, attentions


class IntermediateLayer(nn.Module):

    def __init__(self, hidden_size):
        super(IntermediateLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = self.hidden_size // 2
        self.dense = nn.Linear(self.hidden_size, self.intermediate_size)
        self.activate_fn = nn.GELU()
    
    def forward(self, hidden_states):
        hidden_states = self.activate_fn(self.dense(hidden_states))
        return hidden_states


class IntermediateOutputLayer(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(IntermediateOutputLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = self.hidden_size // 2
        self.dropout_rate = dropout
        self.dense = nn.Linear(self.intermediate_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, hidden_states, inputs):
        hidden_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states + inputs)
        return hidden_states


class FeedForwardLayer(nn.Module):

    def __init__(self, hidden_size, dropout):
        super(FeedForwardLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.intermediate_layer = IntermediateLayer(self.hidden_size)
        self.intermediate_output_layer = IntermediateOutputLayer(self.hidden_size, self.dropout_rate)

    def forward(self, hidden_states):
        hidden_states1 = self.intermediate_layer(hidden_states)
        hidden_states = self.intermediate_output_layer(hidden_states1, hidden_states)
        return hidden_states


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def clone_module(module: nn.Module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


if __name__ == "__main__":
    model = nn.Conv2d(3, 4, kernel_size=(3,))
    model.apply(initialize_weights)

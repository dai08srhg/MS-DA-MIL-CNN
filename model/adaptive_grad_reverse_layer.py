# -*- coding: utf-8 -*-

# Adaptive_DANNの勾配符号反転層

import torch
from torch.autograd import Function


class AdaptiveGradReverse(Function):
    """
    Gradient reverse layer
    lamda: domain adversarial regularization rate
    attention: attention weights
    """
    @staticmethod
    def forward(ctx, x, lamda, attention):
        ctx.lamda = lamda
        ctx.attention = attention
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        attention = ctx.attention.squeeze(0) # [1,100] --> [100]
        max_attention = torch.max(attention)
        adaptive_attention = max_attention-attention
        adaptive_attention = adaptive_attention.unsqueeze(1)
        output = (grad_output.neg() * ctx.lamda)
        adaptive_output = adaptive_attention * output
        return adaptive_output, None, None
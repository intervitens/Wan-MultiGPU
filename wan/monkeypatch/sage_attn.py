import math

import torch
import torch.amp as amp
import torch.nn as nn

from ..modules.attention import sage_attention
from sageattention import sageattn

def wan_selfattn_fwd(self, x, seq_lens, grid_sizes, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        seq_lens(Tensor): Shape [B]
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    x = sage_attention(
        q=rope_apply(q, grid_sizes, freqs),
        k=rope_apply(k, grid_sizes, freqs),
        v=v,
        k_lens=seq_lens,
        window_size=self.window_size)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x



def wan_t2v_ca_fwd(self, x, context, context_lens):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
        context_lens(Tensor): Shape [B]
    """
    b, n, d = x.size(0), self.num_heads, self.head_dim

    # compute query, key, value
    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)

    # compute attention
    x = sage_attention(q, k, v, k_lens=context_lens)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x



def wan_i2v_ca_fwd(self, x, context, context_lens):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
        context_lens(Tensor): Shape [B]
    """
    context_img = context[:, :257]
    context = context[:, 257:]
    b, n, d = x.size(0), self.num_heads, self.head_dim

    # compute query, key, value
    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)
    k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
    v_img = self.v_img(context_img).view(b, -1, n, d)
    img_x = sage_attention(q, k_img, v_img, k_lens=None)
    # compute attention
    x = sage_attention(q, k, v, k_lens=context_lens)

    # output
    x = x.flatten(2)
    img_x = img_x.flatten(2)
    x = x + img_x
    x = self.o(x)
    return x


WAN_CROSSATTENTION_FWD_FN = {
    't2v': wan_t2v_ca_fwd,
    'i2v': wan_i2v_ca_fwd,
}

def monkeypatch_wan_model_sage_attn(model, ca_type):
    for block in model.blocks:
        block.self_attn.forward = wan_selfattn_fwd.__get__(block.self_attn)
        block.cross_attn.forward = WAN_CROSSATTENTION_FWD_FN[ca_type].__get__(block.cross_attn)
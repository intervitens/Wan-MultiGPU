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


def wan_attn_block_fwd(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, 6, C]
        seq_lens(Tensor): Shape [B], length of each sequence in batch
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    assert e.dtype == torch.float32
    with amp.autocast("cuda", dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
    assert e[0].dtype == torch.float32

    # self-attention
    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        freqs)
    with amp.autocast("cuda", dtype=torch.float32):
        x = x + y * e[2]

    # cross-attention & ffn function
    def cross_attn_ffn(x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5]
        return x

    x = cross_attn_ffn(x, context, context_lens, e)
    return x

def monkeypatch_wan_sage_attn(cls):
    cls.WanAttentionBlock.forward = wan_attn_block_fwd
    cls.WanI2VCrossAttention.forward = wan_i2v_ca_fwd
    cls.WanT2VCrossAttention.forward = wan_t2v_ca_fwd
    cls.WanSelfAttention.forward = wan_selfattn_fwd

    return cls


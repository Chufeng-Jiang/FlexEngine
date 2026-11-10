"""Run a complete synthetic Qwen-style decoder without installing transformers.

Independent dense causal reference verifies RMSNorm, RoPE, attention, GQA,
KV page mapping, last-token logits, incremental prefill, and multi-request
isolation. This is a toy architecture test, NOT official Qwen weights parity.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from flexengine.engine.kv_cache.allocator import CacheLayout, PagedKVCache
from flexengine.engine.scheduler.policies import Chunk
from flexengine.inference.models.qwen2 import Qwen2Adapter


class FakeRMS(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        f = x.float()
        return self.weight * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class FakeMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(h, h*2), nn.Linear(h,h*2), nn.Linear(h*2,h)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FakeLayer(nn.Module):
    def __init__(self, h, nq, nkv, dim):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = FakeRMS(h), FakeRMS(h)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(h, nq*dim)
        self.self_attn.k_proj = nn.Linear(h, nkv*dim)
        self.self_attn.v_proj = nn.Linear(h, nkv*dim)
        self.self_attn.o_proj = nn.Linear(nq*dim,h)
        self.mlp = FakeMLP(h)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(model_type="qwen2", hidden_size=32, num_attention_heads=4,
                                    num_key_value_heads=2, head_dim=8, num_hidden_layers=2,
                                    rope_theta=1000000.0, eos_token_id=1)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(50,32)
        self.model.layers = nn.ModuleList([FakeLayer(32,4,2,8) for _ in range(2)])
        self.model.norm = FakeRMS(32)
        self.lm_head = nn.Linear(32,50,bias=False)


def independent_reference(model, ids):
    conf = model.config
    x = model.model.embed_tokens(torch.tensor(ids)).unsqueeze(0)
    nq,nkv,dim = conf.num_attention_heads,conf.num_key_value_heads,conf.head_dim
    positions = torch.arange(len(ids)).float()
    inv = 1.0 / (conf.rope_theta ** (torch.arange(0,dim,2).float()/dim))
    angles = torch.outer(positions,inv)
    cos=torch.cat((angles,angles),-1).cos()[None,None]
    sin=torch.cat((angles,angles),-1).sin()[None,None]
    def rope(t):
        return t*cos+torch.cat((-t[...,dim//2:],t[...,:dim//2]),-1)*sin
    for layer in model.model.layers:
        h=layer.input_layernorm(x)
        a=layer.self_attn
        q=rope(a.q_proj(h).view(1,len(ids),nq,dim).transpose(1,2))
        k=rope(a.k_proj(h).view(1,len(ids),nkv,dim).transpose(1,2))
        v=a.v_proj(h).view(1,len(ids),nkv,dim).transpose(1,2)
        k=k.repeat_interleave(nq//nkv,1)
        v=v.repeat_interleave(nq//nkv,1)
        mask=torch.ones((len(ids),len(ids)),dtype=torch.bool).tril()
        o=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        x=x+a.o_proj(o.transpose(1,2).reshape(1,len(ids),32))
        x=x+layer.mlp(layer.post_attention_layernorm(x))
    return model.lm_head(model.model.norm(x[:, -1, :]))[0]


def test_qwen_synthetic_logits_full_incremental_and_multi():
    torch.manual_seed(19)
    model=FakeQwen().eval()
    adapter=Qwen2Adapter(model,"cpu", attention_backend="blockwise")
    cache=PagedKVCache(CacheLayout(2,40,4,2,8))
    a=[2,5,3,8,9,4,7,12,10]
    b=[16,8,9,2]
    cache.reserve("a",len(a))
    cache.reserve("b",len(b))
    with torch.no_grad():
        got=adapter.forward([Chunk("a",a,0),Chunk("b",b,0)],cache)
        torch.testing.assert_close(got[0],independent_reference(model,a),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(got[1],independent_reference(model,b),atol=1e-5,rtol=1e-5)
        cache.free_request("a")
        cache.reserve("a",len(a))
        adapter.forward([Chunk("a",a[:3],0)],cache)
        adapter.forward([Chunk("a",a[3:8],3)],cache)
        last=adapter.forward([Chunk("a",a[8:],8)],cache)[0]
        torch.testing.assert_close(last,independent_reference(model,a),atol=1e-5,rtol=1e-5)
        cache.assert_invariants()


def test_qwen_accepts_transformers_default_rope_metadata():
    model = FakeQwen().eval()
    model.config.rope_scaling = {"rope_theta": model.config.rope_theta, "rope_type": "default"}

    Qwen2Adapter(model, "cpu")


def test_qwen_reads_theta_from_transformers_default_rope_metadata():
    model = FakeQwen().eval()
    del model.config.rope_theta
    model.config.rope_scaling = {"rope_theta": 1000000.0, "rope_type": "default"}

    adapter = Qwen2Adapter(model, "cpu")

    expected = 1.0 / 1000000.0 ** (torch.arange(0, adapter.dim, 2).float() / adapter.dim)
    torch.testing.assert_close(adapter.inv_freq, expected)


def test_qwen_rejects_actual_rope_scaling():
    model = FakeQwen().eval()
    model.config.rope_scaling = {"factor": 2.0, "rope_type": "linear"}

    with pytest.raises(ValueError, match="RoPE scaling is not implemented"):
        Qwen2Adapter(model, "cpu")

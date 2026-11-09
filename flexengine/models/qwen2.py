"""Own Qwen2.5 decoder execution using HF weights, NOT model.generate().

Validated target model architecture: Qwen2ForCausalLM (0.5/1.5/3B).
Other families, sliding-window attention, Qwen3 q/k norms, MoE, quantized
weights, arbitrary RoPE scaling are not supported.
"""
from __future__ import annotations

import torch

from flexengine.attention.paged import attention_for_request
from flexengine.observability.traces import nvtx


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class Qwen2Adapter:
    def __init__(self, model: torch.nn.Module, device: str, rmsnorm_backend: str = "torch",
                 attention_backend: str = "blockwise"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.config = model.config
        if getattr(self.config, "model_type", None) != "qwen2":
            raise ValueError("Only Qwen2/Qwen2.5 models with model_type=qwen2 are supported")
        if getattr(self.config, "use_sliding_window", False):
            raise ValueError("Sliding-window attention is not implemented")
        rope_scaling = getattr(self.config, "rope_scaling", None)
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type")) if isinstance(rope_scaling, dict) else None
        if rope_scaling and rope_type != "default":
            raise ValueError("RoPE scaling is not implemented")
        self.nq = int(self.config.num_attention_heads)
        self.nkv = int(self.config.num_key_value_heads)
        self.dim = int(getattr(self.config, "head_dim", self.config.hidden_size // self.nq))
        self.layers = int(self.config.num_hidden_layers)
        if self.nq % self.nkv:
            raise ValueError("Invalid GQA head ratio")
        self.rmsnorm_backend = rmsnorm_backend
        self.attention_backend = attention_backend
        theta = getattr(self.config, "rope_theta", None)
        if theta is None and rope_type == "default":
            theta = rope_scaling.get("rope_theta")
        theta = float(theta if theta is not None else 10000.0)
        self.inv_freq = 1.0 / theta ** (torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32) / self.dim)
        self.tokenizer = None  # filled by from_pretrained or supplied externally
        if not hasattr(model, "lm_head") or not hasattr(model, "model"):
            raise ValueError("Expected Qwen2ForCausalLM modules")

    @classmethod
    def from_pretrained(cls, model_id: str, device="cuda:0", dtype="float16", revision: str | None = None, **kwargs) -> Qwen2Adapter:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype_obj = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        try:
            hf = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype_obj, trust_remote_code=False,
                                                      low_cpu_mem_usage=True)
        except TypeError:
            hf = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype_obj, trust_remote_code=False,
                                                      low_cpu_mem_usage=True)
        hf.to(device)
        adapter = cls(hf, device, **kwargs)
        adapter.tokenizer = tokenizer
        return adapter

    def _norm(self, x, layer_norm):
        if self.rmsnorm_backend == "triton":
            from flexengine.kernels.rmsnorm import triton_rmsnorm
            return triton_rmsnorm(x.contiguous(), layer_norm.weight, layer_norm.variance_epsilon)
        return layer_norm(x)

    def _rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        freqs = positions.float().unsqueeze(1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(q.dtype).unsqueeze(1), emb.sin().to(q.dtype).unsqueeze(1)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

    @torch.inference_mode()
    def forward(self, chunks, cache) -> list[torch.Tensor]:
        """chunks: list of objects with request_id, input_ids, start.

        Each request owns independent positions and KV pages; projections/MLP
        process all new tokens in a single matrix batch per layer. Attention
        loops over requests to avoid cross-request leakage.
        """
        lengths = [len(c.input_ids) for c in chunks]
        if not chunks or any(n < 1 for n in lengths):
            raise ValueError("empty forward chunks")
        ids = torch.tensor([t for c in chunks for t in c.input_ids], device=self.device, dtype=torch.long)
        positions = torch.tensor([pos for c in chunks for pos in range(c.start, c.start + len(c.input_ids))],
                                 device=self.device, dtype=torch.long)
        with nvtx(f"executor/embeddings n={len(ids)}"):
            x = self.model.model.embed_tokens(ids)
        for idx, layer in enumerate(self.model.model.layers):
            with nvtx(f"layer/{idx}/qkv"):
                h = self._norm(x, layer.input_layernorm)
                att = layer.self_attn
                q = att.q_proj(h).view(len(ids), self.nq, self.dim)
                k = att.k_proj(h).view(len(ids), self.nkv, self.dim)
                v = att.v_proj(h).view(len(ids), self.nkv, self.dim)
                q, k = self._rope(q, k, positions)
            result = []
            offset = 0
            for c, n in zip(chunks, lengths):
                with nvtx(f"attention/block_table layer={idx} len={c.start+n}"):
                    cache.write(idx, c.request_id, c.start, k[offset : offset + n], v[offset : offset + n])
                    result.append(attention_for_request(q[offset : offset + n], cache, idx, c.request_id,
                                                        c.start + n, positions[offset : offset + n],
                                                        backend=self.attention_backend))
                offset += n
            with nvtx(f"layer/{idx}/mlp"):
                output = torch.cat(result, 0).reshape(len(ids), self.nq * self.dim)
                x = x + att.o_proj(output)
                x = x + layer.mlp(self._norm(x, layer.post_attention_layernorm))
        # Compute vocabulary projection only on the last token of each chunk.
        ends = torch.tensor([sum(lengths[:j + 1]) - 1 for j in range(len(lengths))],
                            device=self.device, dtype=torch.long)
        x_last = self._norm(x.index_select(0, ends), self.model.model.norm)
        logits = self.model.lm_head(x_last)
        return [logits[j] for j in range(len(lengths))]

    @torch.inference_mode()
    def decode_static(self, input_ids, positions, block_tables, active_mask, cache) -> torch.Tensor:
        """Graph-compatible one-token decode over fixed padded batch tensors."""
        batch = input_ids.shape[0]
        if input_ids.ndim != 1 or positions.shape != (batch,) or active_mask.shape != (batch,):
            raise ValueError("invalid static decode inputs")
        block_size = cache.layout.block_size
        max_tokens = block_tables.shape[1] * block_size
        rows = torch.arange(batch, device=self.device)
        active = active_mask.bool()
        x = self.model.model.embed_tokens(input_ids)
        for idx, layer in enumerate(self.model.model.layers):
            h = self._norm(x, layer.input_layernorm)
            att = layer.self_attn
            q = att.q_proj(h).view(batch, self.nq, self.dim)
            k = att.k_proj(h).view(batch, self.nkv, self.dim)
            v = att.v_proj(h).view(batch, self.nkv, self.dim)
            q, k = self._rope(q, k, positions)
            block_ids = block_tables[rows, positions // block_size]
            offsets = positions % block_size
            old_k = cache.k[idx, block_ids, offsets]
            old_v = cache.v[idx, block_ids, offsets]
            cache.k[idx, block_ids, offsets] = torch.where(active[:, None, None], k, old_k)
            cache.v[idx, block_ids, offsets] = torch.where(active[:, None, None], v, old_v)

            if self.attention_backend == "triton":
                from flexengine.kernels.paged_attention import triton_paged_decode_batch
                output = triton_paged_decode_batch(q, cache, idx, block_tables, positions, active_mask)
            elif self.attention_backend == "flashinfer":
                from flexengine.attention.flashinfer_backend import flashinfer_paged_decode_batch
                output = flashinfer_paged_decode_batch(q, cache, idx, block_tables, positions, active_mask)
            else:
                token_positions = torch.arange(max_tokens, device=self.device)
                table_columns = token_positions // block_size
                token_offsets = token_positions % block_size
                pages = block_tables[:, table_columns]
                keys = cache.k[idx, pages, token_offsets]
                values = cache.v[idx, pages, token_offsets]
                keys = keys.repeat_interleave(self.nq // self.nkv, dim=2)
                values = values.repeat_interleave(self.nq // self.nkv, dim=2)
                scores = torch.einsum("bhd,bthd->bht", q.float(), keys.float()) / self.dim**0.5
                valid = (token_positions[None, :] <= positions[:, None]) & active[:, None]
                scores = scores.masked_fill(~valid[:, None, :], -1e4)
                weights = torch.softmax(scores, dim=-1) * active[:, None, None]
                output = torch.einsum("bht,bthd->bhd", weights.to(values.dtype), values)
            x = x + att.o_proj(output.reshape(batch, self.nq * self.dim))
            x = x + layer.mlp(self._norm(x, layer.post_attention_layernorm))
        x = self._norm(x, self.model.model.norm)
        return self.model.lm_head(x) * active[:, None]

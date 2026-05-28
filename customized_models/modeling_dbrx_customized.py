# coding=utf-8
# Copyright 2024 Databricks Mosaic Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DBRX model (customized with forward-pass hooks for MoE-circuits analysis).

This file is a copy of transformers v4.52.0 src/transformers/models/dbrx/modeling_dbrx.py
with hook-injection points added so that the four critical tensors required by
experiments/build_dag.py are exposed in a dict returned alongside the
normal forward output:

    hook_after_res1            — residual stream after the attention residual add,
                                 BEFORE the post-attention layernorm
    hook_after_norm2           — post-attention-layernormed input to the MoE block
    hook_selected_experts      — top-K expert indices from DbrxRouter
    hook_expert_weighted_outputs — per-token, per-K-slot weighted expert FFN outputs

Auxiliary q/k/v/attention-weight hooks are also captured for consistency with the
other customized models in this directory. Only the eager attention path
implements the hook captures; SDPA / FlashAttention paths fall through to the
eager path via DbrxSdpaAttention.forward's existing `output_attentions` redirect.
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring, is_torch_flex_attn_available, logging
from transformers.models.dbrx.configuration_dbrx import DbrxConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask


if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


class DbrxRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    r"""Computes auxiliary load balancing loss as in Switch Transformer."""
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return torch.tensor(0.0)

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class DbrxAttention(nn.Module):
    """Multi-head self attention (eager)."""

    def __init__(self, config: DbrxConfig, block_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_seq_len
        self.block_idx = block_idx
        if block_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `block_idx` is not recommended and will "
                + "lead to errors during the forward call if caching is used. Please make sure to provide a `block_idx` "
                + "when creating this class."
            )

        attn_config = config.attn_config
        self.attn_pdrop = attn_config.attn_pdrop
        self.clip_qkv = attn_config.clip_qkv
        self.num_key_value_heads = attn_config.kv_n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = attn_config.rope_theta
        self.is_causal = True

        self.Wqkv = nn.Linear(
            self.hidden_size, self.hidden_size + 2 * self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = DbrxRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.Wqkv(hidden_states)
        min_val = -self.clip_qkv if self.clip_qkv is not None else None
        max_val = self.clip_qkv
        qkv_states = qkv_states.clamp(min=min_val, max=max_val)

        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=2,
        )

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        #### added — capture pre-RoPE q/k and v for consistency with other customized models
        hook_q = query_states.detach().clone()
        hook_k = key_states.detach().clone()
        hook_v = value_states.detach().clone()
        ########

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.block_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_pdrop, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                + f" {attn_output.size()}"
            )

        #### added
        hook_attn_weights = attn_weights  # [bsz, n_heads, q_len, k_len]
        ########

        attn_output = attn_output.transpose(1, 2).contiguous()
        #### added
        hook_before_matmul_wo = attn_output  # [bsz, q_len, n_heads, head_dim]
        ########
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights_ret = None
        else:
            attn_weights_ret = attn_weights

        #### added — pack 7-tuple analogous to Mixtral's attn_hooks
        attn_hooks = (hook_q, hook_k, hook_v, None, None, hook_attn_weights, hook_before_matmul_wo)
        ########
        return attn_output, attn_weights_ret, past_key_value, attn_hooks


class DbrxFlashAttention2(DbrxAttention):
    """Dbrx flash attention module.

    NOTE: customized DBRX is only intended for use with `attn_implementation="eager"`.
    Flash attention is kept structurally for compatibility but does not produce hooks;
    callers that need hooks must load with `attn_implementation="eager"`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        raise NotImplementedError(
            "DbrxFlashAttention2 is not supported in the customized DBRX model. "
            "Load with `attn_implementation='eager'`."
        )


class DbrxSdpaAttention(DbrxAttention):
    """SDPA path — falls back to eager so hooks are captured."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        # Customized DBRX requires the eager path to capture hooks.
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


DBRX_ATTENTION_CLASSES = {
    "eager": DbrxAttention,
    "flash_attention_2": DbrxFlashAttention2,
    "sdpa": DbrxSdpaAttention,
}


class DbrxNormAttentionNorm(nn.Module):
    def __init__(self, config: DbrxConfig, block_idx: Optional[int] = None):
        super().__init__()
        self.block_idx = block_idx
        self.resid_pdrop = config.resid_pdrop
        self.norm_1 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = DBRX_ATTENTION_CLASSES[config._attn_implementation](
            config=config,
            block_idx=block_idx,
        )
        self.norm_2 = nn.LayerNorm(config.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[Tuple[torch.Tensor]]]:
        residual_states = hidden_states
        #### added — input to the block, before any norm/attention
        hook_layer_input = hidden_states
        ########
        hidden_states = self.norm_1(hidden_states).to(hidden_states.dtype)

        hidden_states, attn_weights, past_key_value, attn_hooks = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        #### added — attention output before residual add and before resid_pdrop
        hook_attn_output = hidden_states
        ########

        hidden_states = nn.functional.dropout(hidden_states, p=self.resid_pdrop, training=self.training)
        hidden_states = hidden_states + residual_states

        residual_states = hidden_states
        #### added — residual stream AFTER attention residual add, BEFORE the post-attention layernorm
        hook_after_res1 = hidden_states
        ########
        hidden_states = self.norm_2(hidden_states).to(hidden_states.dtype)
        #### added — post-attention-layernormed input to the MoE block
        hook_after_norm2 = hidden_states
        ########

        #### added — pack 4-tuple of layer-level hooks captured so far
        norm_attn_norm_hooks = (hook_layer_input, hook_attn_output, hook_after_res1, hook_after_norm2)
        ########
        return residual_states, hidden_states, attn_weights, past_key_value, norm_attn_norm_hooks, attn_hooks


class DbrxRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_jitter_eps: Optional[float],
        moe_normalize_expert_weights: Optional[float],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_normalize_expert_weights = moe_normalize_expert_weights

        self.layer = nn.Linear(self.hidden_size, self.moe_num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.training and self.moe_jitter_eps is not None:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.moe_jitter_eps, 1.0 + self.moe_jitter_eps
            )
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        weights = self.layer(hidden_states).softmax(dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)
        #### added — capture top-K indices immediately after topk
        hook_selected_experts = top_experts
        ########

        top_weights_scale = (
            torch.norm(top_weights, p=self.moe_normalize_expert_weights, dim=-1, keepdim=True)
            if self.moe_normalize_expert_weights is not None
            else 1.0
        )
        top_weights = top_weights / top_weights_scale

        weights = weights.to(hidden_states.dtype)
        top_weights = top_weights.to(hidden_states.dtype)
        #### added — also capture the (post-normalization) routing weights for analysis
        hook_routing_weights = top_weights
        router_hooks = (hook_selected_experts, hook_routing_weights)
        ########
        return weights, top_weights, top_experts, router_hooks


class DbrxExpertGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, ffn_act_fn: dict):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts

        self.w1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.v1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))

        act_fn_name = ffn_act_fn.get("name", "silu")
        self.activation_fn = ACT2FN[act_fn_name]

    def forward(
        self, x: torch.Tensor, expert_w1: torch.Tensor, expert_v1: torch.Tensor, expert_w2: torch.Tensor
    ) -> torch.Tensor:
        gate_proj = x.matmul(expert_w1.t())
        up_proj = x.matmul(expert_v1.t())
        gate_proj = self.activation_fn(gate_proj)
        intermediate_states = gate_proj * up_proj
        down_proj = intermediate_states.matmul(expert_w2)
        return down_proj


class DbrxExperts(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, moe_top_k: int, ffn_act_fn: dict):
        super().__init__()
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k  #### added — needed to pre-allocate the per-token, per-K-slot hook tensor
        self.hidden_size = hidden_size  #### added
        self.mlp = DbrxExpertGLU(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            moe_num_experts=moe_num_experts,
            ffn_act_fn=ffn_act_fn,
        )

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, top_weights: torch.Tensor, top_experts: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)
        #### added — per-token, per-K-slot weighted expert output buffer
        hook_expert_weighted_outputs = torch.empty(
            (bsz * q_len, self.moe_top_k, hidden_size), dtype=x.dtype, device=x.device
        )
        ########

        expert_mask = nn.functional.one_hot(top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        # Chunk experts at once to avoid storing full parameter multiple times in autograd
        w1_chunked = self.mlp.w1.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
            self.moe_num_experts, dim=0
        )
        v1_chunked = self.mlp.v1.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
            self.moe_num_experts, dim=0
        )
        w2_chunked = self.mlp.w2.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
            self.moe_num_experts, dim=0
        )
        w1_chunked = [w1.squeeze(dim=0) for w1 in w1_chunked]
        v1_chunked = [v1.squeeze(dim=0) for v1 in v1_chunked]
        w2_chunked = [w2.squeeze(dim=0) for w2 in w2_chunked]
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx
            topk_list = topk_idx

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = (
                self.mlp(expert_tokens, w1_chunked[expert_idx], v1_chunked[expert_idx], w2_chunked[expert_idx])
                * top_weights[token_list, topk_list, None]
            )
            #### added — store the per-(token, k-slot) weighted expert contribution
            hook_expert_weighted_outputs[token_list, topk_list] = expert_out.to(hook_expert_weighted_outputs.dtype)
            ########

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out, hook_expert_weighted_outputs


class DbrxFFN(nn.Module):
    def __init__(self, config: DbrxConfig):
        super().__init__()

        ffn_config = config.ffn_config
        self.router = DbrxRouter(
            hidden_size=config.d_model,
            moe_num_experts=ffn_config.moe_num_experts,
            moe_top_k=ffn_config.moe_top_k,
            moe_jitter_eps=ffn_config.moe_jitter_eps,
            moe_normalize_expert_weights=ffn_config.moe_normalize_expert_weights,
        )

        self.experts = DbrxExperts(
            hidden_size=config.d_model,
            ffn_hidden_size=ffn_config.ffn_hidden_size,
            moe_num_experts=ffn_config.moe_num_experts,
            moe_top_k=ffn_config.moe_top_k,  #### added
            ffn_act_fn=ffn_config.ffn_act_fn,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        weights, top_weights, top_experts, router_hooks = self.router(x)
        out, hook_expert_weighted_outputs = self.experts(x, weights, top_weights, top_experts)
        #### added — full mlp-level hooks tuple:
        #   (router_logits, hook_selected_experts, hook_expert_weighted_outputs, hook_routing_weights)
        hook_selected_experts, hook_routing_weights = router_hooks
        mlp_hooks = (weights, hook_selected_experts, hook_expert_weighted_outputs, hook_routing_weights)
        ########
        return out, weights, mlp_hooks


class DbrxBlock(nn.Module):
    def __init__(self, config: DbrxConfig, block_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.block_idx = block_idx
        self.norm_attn_norm = DbrxNormAttentionNorm(
            config=config,
            block_idx=block_idx,
        )
        self.ffn = DbrxFFN(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple:
        # Norm + Attention + Norm
        resid_states, hidden_states, self_attn_weights, present_key_value, norm_attn_norm_hooks, attn_hooks = self.norm_attn_norm(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Fully Connected
        hidden_states, router_logits, mlp_hooks = self.ffn(hidden_states)
        #### added — MoE output BEFORE residual dropout and residual add
        hook_mlp_output = hidden_states
        ########
        hidden_states = nn.functional.dropout(hidden_states, p=self.resid_pdrop, training=self.training)
        hidden_states = resid_states + hidden_states
        #### added — final output of the decoder block
        hook_layer_output = hidden_states
        ########

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        #### added — assemble per-layer hook tuple to be unpacked in DbrxModel.forward
        # layer_hooks = (hook_layer_input, hook_attn_output, hook_after_res1, hook_after_norm2, hook_mlp_output, hook_layer_output)
        hook_layer_input, hook_attn_output, hook_after_res1, hook_after_norm2 = norm_attn_norm_hooks
        layer_hooks = (hook_layer_input, hook_attn_output, hook_after_res1, hook_after_norm2, hook_mlp_output, hook_layer_output)
        my_hooks = (*layer_hooks, *attn_hooks, *mlp_hooks)
        outputs += (my_hooks,)
        ########
        return outputs


@auto_docstring
class DbrxPreTrainedModel(PreTrainedModel):
    config_class = DbrxConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DbrxBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DbrxExpertGLU):
            module.w1.data.normal_(mean=0.0, std=std)
            module.v1.data.normal_(mean=0.0, std=std)
            module.w2.data.normal_(mean=0.0, std=std)


@auto_docstring
class DbrxModel(DbrxPreTrainedModel):
    """Transformer decoder consisting of *config.n_layers* DbrxBlock layers."""

    def __init__(self, config: DbrxConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.emb_pdrop = config.emb_pdrop

        self.wte = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.blocks = nn.ModuleList([DbrxBlock(config, block_idx) for block_idx in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, bias=False)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding):
        self.wte = value

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[Union[Tuple, MoeModelOutputWithPast], dict]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.emb_pdrop, training=self.training)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        #### added — pre-allocate per-layer hook tensors (no explicit device; picks up torch.set_default_device)
        layer_counter = 0
        dict_pos = -1  # the my_hooks tuple is always the last element of layer outputs
        n_layers = self.config.n_layers
        batch_size = hidden_states.shape[0]
        n_tokens = hidden_states.shape[1]
        n_dim = hidden_states.shape[2]
        n_heads = self.config.n_heads
        n_kv_heads = self.config.attn_config.kv_n_heads
        n_head_dim = n_dim // n_heads
        n_experts = self.config.ffn_config.moe_num_experts
        top_k = self.config.ffn_config.moe_top_k
        hook_layer_input = torch.empty((n_layers, batch_size, n_tokens, n_dim))
        hook_attn_output = torch.empty((n_layers, batch_size, n_tokens, n_dim))
        hook_after_res1 = torch.empty((n_layers, batch_size, n_tokens, n_dim))
        hook_after_norm2 = torch.empty((n_layers, batch_size, n_tokens, n_dim))
        hook_mlp_output = torch.empty((n_layers, batch_size, n_tokens, n_dim))
        hook_layer_output = torch.empty((n_layers, batch_size, n_tokens, n_dim))
        hook_q = torch.empty((n_layers, batch_size, n_heads, n_tokens, n_head_dim))
        hook_k = torch.empty((n_layers, batch_size, n_kv_heads, n_tokens, n_head_dim))
        hook_v = torch.empty((n_layers, batch_size, n_kv_heads, n_tokens, n_head_dim))
        hook_attn_weights = torch.empty((n_layers, batch_size, n_heads, n_tokens, n_tokens))
        hook_before_matmul_wo = torch.empty((n_layers, batch_size, n_tokens, n_heads, n_head_dim))
        router_logits = torch.empty((n_layers, batch_size * n_tokens, n_experts))
        hook_selected_experts = torch.empty((n_layers, batch_size * n_tokens, top_k))
        hook_expert_weighted_outputs = torch.empty((n_layers, batch_size * n_tokens, top_k, n_dim))
        hook_routing_weights = torch.empty((n_layers, batch_size * n_tokens, top_k))
        ########

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                block_outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = block_outputs[0]

            if use_cache:
                next_decoder_cache = block_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (block_outputs[1],)

            if output_router_logits:
                # block_outputs is: (hidden_states, [self_attn_weights], [present_kv], [router_logits], my_hooks)
                # router_logits is at index -2 because my_hooks is last
                all_router_logits += (block_outputs[-2],)

            #### added — unpack my_hooks (always the last element) into per-layer buffers
            my_hooks = block_outputs[dict_pos]
            hook_layer_input[layer_counter] = my_hooks[0]
            hook_attn_output[layer_counter] = my_hooks[1]
            hook_after_res1[layer_counter] = my_hooks[2]
            hook_after_norm2[layer_counter] = my_hooks[3]
            hook_mlp_output[layer_counter] = my_hooks[4]
            hook_layer_output[layer_counter] = my_hooks[5]
            hook_q[layer_counter] = my_hooks[6]
            hook_k[layer_counter] = my_hooks[7]
            hook_v[layer_counter] = my_hooks[8]
            # my_hooks[9], my_hooks[10] are None placeholders (parity with Mixtral's attn_hooks layout)
            hook_attn_weights[layer_counter] = my_hooks[11]
            hook_before_matmul_wo[layer_counter] = my_hooks[12]
            router_logits[layer_counter] = my_hooks[13]
            hook_selected_experts[layer_counter] = my_hooks[14]
            hook_expert_weighted_outputs[layer_counter] = my_hooks[15]
            hook_routing_weights[layer_counter] = my_hooks[16]
            layer_counter += 1
            ########

        #### added — reshape/transpose so the leading dim becomes batch (matches Mixtral conventions)
        hook_layer_input = hook_layer_input.transpose(0, 1)
        hook_attn_output = hook_attn_output.transpose(0, 1)
        hook_after_res1 = hook_after_res1.transpose(0, 1)
        hook_after_norm2 = hook_after_norm2.transpose(0, 1)
        hook_mlp_output = hook_mlp_output.transpose(0, 1)
        hook_layer_output = hook_layer_output.transpose(0, 1)
        hook_q = hook_q.transpose(0, 1)
        hook_k = hook_k.transpose(0, 1)
        hook_v = hook_v.transpose(0, 1)
        hook_attn_weights = hook_attn_weights.transpose(0, 1)
        hook_before_matmul_wo = hook_before_matmul_wo.permute(1, 0, 3, 2, 4)
        router_logits = router_logits.transpose(0, 1)
        hook_selected_experts = hook_selected_experts.reshape(n_layers, batch_size, n_tokens, top_k).transpose(0, 1)
        hook_expert_weighted_outputs = hook_expert_weighted_outputs.reshape(
            n_layers, batch_size, n_tokens, top_k, n_dim
        ).transpose(0, 1)
        hook_routing_weights = hook_routing_weights.reshape(n_layers, batch_size, n_tokens, top_k).transpose(0, 1)
        hook_dict = {
            "hook_layer_input": hook_layer_input, "hook_attn_output": hook_attn_output,
            "hook_after_res1": hook_after_res1, "hook_after_norm2": hook_after_norm2,
            "hook_mlp_output": hook_mlp_output, "hook_layer_output": hook_layer_output,
            "hook_q": hook_q, "hook_k": hook_k, "hook_v": hook_v,
            "hook_attn_weights": hook_attn_weights,
            "hook_before_matmul_wo": hook_before_matmul_wo, "router_logits": router_logits,
            "hook_selected_experts": hook_selected_experts,
            "hook_expert_weighted_outputs": hook_expert_weighted_outputs,
            "hook_routing_weights": hook_routing_weights,
        }
        ########

        hidden_states = self.norm_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            output = tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
            return output, hook_dict  #### modified
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        ), hook_dict  #### modified

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


@auto_docstring(
    custom_intro="""
    The DBRX Model transformer for causal language modeling (customized with hooks).
    """
)
class DbrxForCausalLM(DbrxPreTrainedModel, GenerationMixin):
    def __init__(self, config: DbrxConfig):
        super().__init__(config)
        self.transformer = DbrxModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.moe_loss_weight = config.ffn_config.moe_loss_weight
        self.num_experts = config.ffn_config.moe_num_experts
        self.num_experts_per_tok = config.ffn_config.moe_top_k

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: DbrxModel):
        self.transformer = decoder

    def get_decoder(self) -> DbrxModel:
        return self.transformer

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[Union[Tuple, MoeCausalLMOutputWithPast], dict]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, hook_dict = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if return_dict:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        # No upscaling to float was ever done for Dbrx
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None and loss is not None:
                loss += self.moe_loss_weight * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return ((loss,) + output if loss is not None else output), hook_dict  #### modified

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        ), hook_dict  #### modified


__all__ = ["DbrxForCausalLM", "DbrxModel", "DbrxPreTrainedModel"]

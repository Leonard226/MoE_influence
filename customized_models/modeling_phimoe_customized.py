# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Phimoe model — customized with forward-pass hook injection.

Derived from transformers v4.52.0 src/transformers/models/phimoe/modeling_phimoe.py.
Hooks mirror the conventions in modeling_mixtral_customized.py: the model's
forward returns (output, hook_dict) and propagates that through ForCausalLM.

Computation is numerically identical to the un-customized model — only hook
captures, pre-allocation, and tuple plumbing are added.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from transformers.models.phimoe.configuration_phimoe import PhimoeConfig


if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
_prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

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


class PhimoeRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Optional[PhimoeConfig] = None,
    ):
        super().__init__()

        self.config = config
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            self.short_mscale = config.rope_scaling.get("short_mscale")
            self.long_mscale = config.rope_scaling.get("long_mscale")
        else:
            self.rope_type = "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

    def forward(self, x, seq_len=None):
        mscale = None
        if self.config.rope_scaling and seq_len:
            mscale = (
                self.long_mscale
                if seq_len > self.config.rope_scaling["original_max_position_embeddings"]
                else self.short_mscale
            )
        inv_freq, attention_scaling = self.rope_init_fn(self.config, x.device, seq_len)
        mscale = attention_scaling if mscale is None else mscale
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(x.dtype), (emb.sin() * mscale).to(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PhimoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: PhimoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.config.attention_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        set_patch=None,  #### added
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        #### added: capture pre-RoPE q/k/v and apply optional patches
        hook_q, hook_k, hook_v = query_states.detach().clone(), key_states.detach().clone(), value_states.detach().clone()
        if set_patch is not None:
            for p in set_patch:
                match p[0]:
                    case 'q':
                        query_states = p[2]
                    case 'k':
                        key_states = p[2]
                    case 'v':
                        value_states = p[2]
        ########

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        #### added
        hook_attn_weights = attn_weights  # [bsz, num_heads, q_len, k_len]
        hook_before_matmul_wo = attn_output  # [bsz, q_len, num_heads, head_dim]
        ########

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights_ret = None
        else:
            attn_weights_ret = attn_weights

        #### added: attn_hooks tuple matches the Mixtral layout
        attn_hooks = (hook_q, hook_k, hook_v, None, None, hook_attn_weights, hook_before_matmul_wo)
        return attn_output, attn_weights_ret, past_key_value, attn_hooks
        ########


class PhimoeFlashAttention2(PhimoeAttention):
    """Flash-attn variant — hooks NOT injected. We always call with eager."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        set_patch=None,
    ):
        raise NotImplementedError(
            "PhimoeFlashAttention2 is not supported in the customized model. "
            "Load with attn_implementation='eager'."
        )


class PhimoeSdpaAttention(PhimoeAttention):
    """SDPA variant — hooks NOT injected. We always call with eager."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        set_patch=None,
    ):
        # Fall back to eager to ensure hooks are populated.
        return PhimoeAttention.forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            set_patch=set_patch,
        )


PHIMOE_ATTENTION_CLASSES = {
    "eager": PhimoeAttention,
    "flash_attention_2": PhimoeFlashAttention2,
    "sdpa": PhimoeSdpaAttention,
}


class PhimoeBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: PhimoeConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MultiplierProcessor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        multiplier: torch.Tensor,
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
        ctx,
        grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expanded = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expanded.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expanded,
            None,
            None,
            None,
            None,
        )


def sparsemixer(scores, jitter_eps, training, top_k=2):
    """Sparse mixer routing — preserves PhiMoE-specific jitter + Heun's method routing.

    Computation kept verbatim from upstream; only top_k==2 is supported.
    """
    if top_k != 2:
        raise ValueError("top_k must be equal to 2")

    # first expert
    with torch.no_grad():
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))
    if training:
        selected_experts = (
            (
                masked_gates
                - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            .max(dim=-1)[1]
            .unsqueeze(-1)
        )
    else:
        selected_experts = max_ind

    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    if training:
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            selected_experts == max_ind,
            torch.rand_like(max_scores) > 0.75,
        )
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

        multiplier = MultiplierProcessor.apply(
            scores,
            multiplier_o,
            selected_experts,
            masked_gates,
            mask_for_one,
        )
    else:
        multiplier = multiplier_o

    # Masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float("-inf"),
    )
    with torch.no_grad():
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))
    if training:
        selected_experts_top2 = (
            (
                masked_gates_top2
                - torch.empty_like(masked_gates_top2, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            .max(dim=-1)[1]
            .unsqueeze(-1)
        )
    else:
        selected_experts_top2 = max_ind
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    if training:
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75,
        )
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

        multiplier_top2 = MultiplierProcessor.apply(
            scores,
            multiplier_top2_o,
            selected_experts_top2,
            masked_gates_top2,
            mask_for_one_top2,
        )
    else:
        multiplier_top2 = multiplier_top2_o

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )


class PhimoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([PhimoeBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters — PhiMoE has BOTH router_jitter_noise and input_jitter_noise.
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = sparsemixer(
            router_logits,
            jitter_eps=self.router_jitter_noise,
            training=self.training,
        )
        hook_selected_experts = selected_experts  #### added
        hook_routing_weights = routing_weights  #### added

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        #### added: pre-alloc per-expert weighted-output capture
        hook_expert_weighted_outputs = torch.zeros(
            (batch_size * sequence_length, self.top_k, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        ########

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            #### added: stash the weighted output for this expert at (top_x, idx)
            hook_expert_weighted_outputs[top_x, idx] = current_hidden_states.to(hook_expert_weighted_outputs.dtype)
            ########

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        mlp_hooks = (router_logits, hook_selected_experts, hook_expert_weighted_outputs, hook_routing_weights)  #### added
        return final_hidden_states, router_logits, mlp_hooks  #### modified


class PhimoeDecoderLayer(nn.Module):
    def __init__(self, config: PhimoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PHIMOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.block_sparse_moe = PhimoeSparseMoeBlock(config)
        # PhiMoE uses LayerNorm (with bias), not RMSNorm.
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        set_patch=None,  #### added
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hook_layer_input = hidden_states  #### added

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, attn_hooks = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            set_patch=set_patch,  #### added
        )
        hook_attn_output = hidden_states  #### added
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hook_after_res1 = hidden_states  #### added
        hidden_states = self.post_attention_layernorm(hidden_states)
        hook_after_norm2 = hidden_states  #### added
        hidden_states, router_logits, mlp_hooks = self.block_sparse_moe(hidden_states)  #### modified
        hook_mlp_output = hidden_states  #### added
        hidden_states = residual + hidden_states
        hook_layer_output = hidden_states  #### added

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        #### added
        layer_hooks = (hook_layer_input, hook_attn_output, hook_after_res1, hook_after_norm2, hook_mlp_output, hook_layer_output)
        my_hooks = (*layer_hooks, *attn_hooks, *mlp_hooks,)
        outputs += (my_hooks,)
        ########
        return outputs


@auto_docstring
class PhimoePreTrainedModel(PreTrainedModel):
    config_class = PhimoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PhimoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False  # MoE models don't work with torch.compile

    def _init_weights(self, module):
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
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@auto_docstring
class PhimoeModel(PhimoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhimoeDecoderLayer`]
    """

    def __init__(self, config: PhimoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PhimoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.rotary_emb = PhimoeRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        patching=None,  #### added
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, seq_len=cache_position[-1] + 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        #### added: pre-allocate hook tensors on default device (picks up torch.set_default_device)
        layer_counter = 0
        n_layers = self.config.num_hidden_layers
        batch_size = hidden_states.shape[0]
        n_tokens = hidden_states.shape[1]
        n_dim = hidden_states.shape[2]
        n_heads = self.config.num_attention_heads
        n_kv_heads = self.config.num_key_value_heads
        n_head_dim = n_dim // n_heads
        n_experts = self.config.num_local_experts
        top_k = self.config.num_experts_per_tok
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
        router_logits_all = torch.empty((n_layers, batch_size * n_tokens, n_experts))
        hook_selected_experts = torch.empty((n_layers, batch_size * n_tokens, top_k))
        hook_expert_weighted_outputs = torch.empty((n_layers, batch_size * n_tokens, top_k, n_dim))
        hook_routing_weights = torch.empty((n_layers, batch_size * n_tokens, top_k))
        ########

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            #### added: assemble per-layer set_patch
            if patching is not None:
                set_patch = []
                for p in patching:
                    if layer_counter == p[1]:
                        set_patch.append(p)
                if len(set_patch) == 0:
                    set_patch = None
            else:
                set_patch = None
            ########

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    set_patch=set_patch,  #### added
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-2],)  #### shifted by 1: hooks are last

            #### added: my_hooks is always last
            my_hooks = layer_outputs[-1]
            hook_layer_input[layer_counter] = my_hooks[0]
            hook_attn_output[layer_counter] = my_hooks[1]
            hook_after_res1[layer_counter] = my_hooks[2]
            hook_after_norm2[layer_counter] = my_hooks[3]
            hook_mlp_output[layer_counter] = my_hooks[4]
            hook_layer_output[layer_counter] = my_hooks[5]
            hook_q[layer_counter] = my_hooks[6]
            hook_k[layer_counter] = my_hooks[7]
            hook_v[layer_counter] = my_hooks[8]
            # my_hooks[9], my_hooks[10] are None placeholders (q2/k2 slots, unused for PhiMoE)
            hook_attn_weights[layer_counter] = my_hooks[11]
            hook_before_matmul_wo[layer_counter] = my_hooks[12]
            router_logits_all[layer_counter] = my_hooks[13]
            hook_selected_experts[layer_counter] = my_hooks[14]
            hook_expert_weighted_outputs[layer_counter] = my_hooks[15]
            hook_routing_weights[layer_counter] = my_hooks[16]
            layer_counter += 1
            ########

        #### added: transpose to [batch, layers, ...]
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
        router_logits_all = router_logits_all.transpose(0, 1)
        hook_selected_experts = hook_selected_experts.reshape(n_layers, batch_size, n_tokens, top_k).transpose(0, 1)
        hook_expert_weighted_outputs = hook_expert_weighted_outputs.reshape(
            n_layers, batch_size, n_tokens, top_k, n_dim
        ).transpose(0, 1)
        hook_routing_weights = hook_routing_weights.transpose(0, 1)
        hook_dict = {
            "hook_layer_input": hook_layer_input,
            "hook_attn_output": hook_attn_output,
            "hook_after_res1": hook_after_res1,
            "hook_after_norm2": hook_after_norm2,
            "hook_mlp_output": hook_mlp_output,
            "hook_layer_output": hook_layer_output,
            "hook_q": hook_q,
            "hook_k": hook_k,
            "hook_v": hook_v,
            "hook_attn_weights": hook_attn_weights,
            "hook_before_matmul_wo": hook_before_matmul_wo,
            "router_logits": router_logits_all,
            "hook_selected_experts": hook_selected_experts,
            "hook_expert_weighted_outputs": hook_expert_weighted_outputs,
            "hook_routing_weights": hook_routing_weights,
        }
        ########

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        ), hook_dict  #### modified

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Phimoe. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
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
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
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
        config: PhimoeConfig,
        past_key_values: Cache,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            text_config = config.get_text_config()
            if getattr(text_config, "use_sliding_window", True) and text_config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - text_config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class PhimoeForCausalLM(PhimoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = PhimoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=self.config.lm_head_bias)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        patching=None,  #### added
        **loss_kwargs,
    ) -> MoeCausalLMOutputWithPast:
        if (
            use_cache
            and self.config.rope_scaling
            and cache_position is not None
            and cache_position[0] == self.config.original_max_position_embeddings
        ):
            logger.warning(
                f"If you are not using the generate method, you may encounter nonsensical outputs after the {self.config.original_max_position_embeddings}th token, as the KV cache needs to be recomputed."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs, hook_dict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            patching=patching,  #### added
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        ), hook_dict  #### modified

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        if (
            past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return model_inputs


# Aliases using the "PhiMoE" capitalization convention requested in spec.
PhiMoEConfig = PhimoeConfig
PhiMoEBlockSparseTop2MLP = PhimoeBlockSparseTop2MLP
PhiMoESparseMoeBlock = PhimoeSparseMoeBlock
PhiMoEAttention = PhimoeAttention
PhiMoEDecoderLayer = PhimoeDecoderLayer
PhiMoEPreTrainedModel = PhimoePreTrainedModel
PhiMoEModel = PhimoeModel
PhiMoEForCausalLM = PhimoeForCausalLM


__all__ = [
    "PhimoePreTrainedModel",
    "PhimoeModel",
    "PhimoeForCausalLM",
    "PhiMoEPreTrainedModel",
    "PhiMoEModel",
    "PhiMoEForCausalLM",
]

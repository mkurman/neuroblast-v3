# coding=utf-8
# Copyright 2025 Mariusz Kurman, MedIT Solutions Sp. z o.o, Poland. All rights reserved.
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

"""PyTorch NeuroBLAST model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import GenerationMixin
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_neuroblast import NeuroBLASTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "NeuroBLASTConfig"


class NeuroBLASTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class NeuroBLASTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(q.dtype)


class NeuroBLASTAttention(nn.Module):
    def __init__(self, config: NeuroBLASTConfig, layer_idx: Optional[int] = None, use_rope: bool = True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.q_norm = NeuroBLASTRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = NeuroBLASTRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The `layer_idx` should be defined when calling the forward function of {self.__class__.__name__}. "
                    "Please make sure to pass a `layer_idx` when creating this class."
                )
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        if self.use_rope and position_embeddings is not None:
             cos, sin = position_embeddings
             cos = cos.squeeze(2)
             sin = sin.squeeze(2)
             query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
        else:
            cos = None
            sin = None

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs={"cos": cos, "sin": sin})

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NeuroBLASTRMSNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        # x: (B, C, H, W)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.view(1, -1, 1, 1) * x_norm.to(input_dtype)


class NeuroBLASTCausalConv2DBlock(nn.Module):
    def __init__(self, config, dilation=1, layer_idx=0):
        super().__init__()
        self.config = config
        self.dilation = dilation
        self.layer_idx = layer_idx
        
        k = config.kernel_size
        d = config.hidden_size
        s = config.scale
        
        self.conv_padding = (k // 2, 0)
        
        if s == 1:
            self.conv = nn.Conv2d(
                d, d,
                kernel_size=(k, k),
                dilation=(1, dilation),
                padding=self.conv_padding,
                bias=False
            )
            self.use_gating = False
            self.use_projection = False
        elif s > 1:
            internal_dim = int(d * s)
            self.conv = nn.Conv2d(
                d, internal_dim,
                kernel_size=(k, k),
                dilation=(1, dilation),
                padding=self.conv_padding,
                bias=False
            )
            self.use_gating = True
            self.use_projection = False
        else:
            internal_dim = max(int(d * s), d // 4)
            self.conv = nn.Conv2d(
                d, internal_dim,
                kernel_size=(k, k),
                dilation=(1, dilation),
                padding=self.conv_padding,
                bias=False
            )
            self.use_gating = False
            self.use_projection = True
            self.proj_back = nn.Conv2d(internal_dim, d, kernel_size=(1, 1), bias=False)

        self.norm_in = NeuroBLASTRMSNorm2d(d, eps=config.rms_norm_eps)
        self.norm_out = NeuroBLASTRMSNorm2d(d, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        residual = x
        y = self.norm_in(x)
        
        k = self.config.kernel_size
        pad_w = (k - 1) * self.dilation
        
        # Pad W on the left
        y_pad = F.pad(y, (pad_w, 0, 0, 0))
        
        y = self.conv(y_pad)
        
        if self.use_gating:
            gate, val = torch.chunk(y, 2, dim=1)
            y = val * F.softmax(gate, dim=1)
        elif self.use_projection:
            y = self.proj_back(y)
            
        y = self.norm_out(y)
        
        x = residual + self.dropout(y)
        return x


class NeuroBLASTDecoderLayer(nn.Module):
    def __init__(self, config: NeuroBLASTConfig, layer_idx: int, attention_type: str = "full_attention"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuroBLASTAttention(
            config=config,
            layer_idx=layer_idx,
            use_rope=(attention_type != "no_rope"),
        )
        self.mlp = NeuroBLASTMLP(config)
        self.input_layernorm = NeuroBLASTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NeuroBLASTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class NeuroBLASTToken2D(nn.Module):
    def forward(self, x, mode="seq_to_2d"):
        if mode == "seq_to_2d":
            return x.permute(0, 2, 1).unsqueeze(2)
        else:
            return x.squeeze(2).permute(0, 2, 1)


class NeuroBLASTRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: (B, L, H, D) or similar. Not used for shape here.
        # position_ids: (B, L)
        
        inv_freq_expanded = self.inv_freq[None, :, None]
        position_ids_expanded = position_ids[:, :, None].float()
        freqs = torch.matmul(position_ids_expanded, self.inv_freq[None, None, :])
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # Output: (B, L, 1, D)
        return cos[:, :, None, :], sin[:, :, None, :]


class NeuroBLASTPreTrainedModel(PreTrainedModel):
    config_class = NeuroBLASTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NeuroBLASTDecoderLayer", "NeuroBLASTCausalConv2DBlock"]
    _skip_keys_device_placement = "past_key_values"

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


class NeuroBLASTModel(NeuroBLASTPreTrainedModel):
    def __init__(self, config: NeuroBLASTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.token2d = NeuroBLASTToken2D()
        
        self.sensory_layers = nn.ModuleList()
        dilatation_step = 1
        for i in range(config.num_sensory_layers):
            if i % 2 == 0:
                layer = NeuroBLASTDecoderLayer(config, layer_idx=i, attention_type="full_attention")
            else:
                dilation = min(2 ** ((i - 1) // dilatation_step), 8)
                layer = NeuroBLASTCausalConv2DBlock(config, dilation=dilation, layer_idx=i)
            self.sensory_layers.append(layer)
            
        self.sensory_to_associative = NeuroBLASTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.associative_layers = nn.ModuleList()
        next_layer_type = "full_attention"
        for i in range(config.num_associative_layers):
            idx = i + config.num_sensory_layers
            if i % 2 == 0:
                layer = NeuroBLASTDecoderLayer(config, layer_idx=idx, attention_type=next_layer_type)
                if next_layer_type == "full_attention":
                    next_layer_type = "no_rope"
                else:
                    next_layer_type = "full_attention"
            else:
                dilation = min(2 ** ((i - 1) // dilatation_step), 8)
                layer = NeuroBLASTCausalConv2DBlock(config, dilation=dilation, layer_idx=idx)
            self.associative_layers.append(layer)
            
        self.sensory_to_motor = NeuroBLASTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.motor_layers = nn.ModuleList()
        next_layer_type = "full_attention"
        for i in range(config.num_motor_layers):
            idx = i + config.num_sensory_layers + config.num_associative_layers
            layer = NeuroBLASTDecoderLayer(config, layer_idx=idx, attention_type=next_layer_type)
            if next_layer_type == "full_attention":
                next_layer_type = "no_rope"
            else:
                next_layer_type = "full_attention"
            self.motor_layers.append(layer)
            
        self.norm = NeuroBLASTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = NeuroBLASTRotaryEmbedding(config)
        
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            past_length = 0
            if past_key_values is not None:
                if isinstance(past_key_values, DynamicCache):
                    past_length = past_key_values.get_seq_length()
                elif isinstance(past_key_values, (tuple, list)):
                    past_length = past_key_values[0][0].shape[-2]
            
            position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
            
        # Create causal mask
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.full((seq_length, seq_length), min_dtype, device=inputs_embeds.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :] # (1, 1, L, L)
        
        # Expand attention_mask
        # attention_mask: (B, L) -> (B, 1, 1, L)
        padding_mask = attention_mask[:, None, None, :].to(inputs_embeds.dtype)
        padding_mask = (1.0 - padding_mask) * min_dtype
        
        combined_mask = causal_mask + padding_mask
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        hidden_states = inputs_embeds
        
        # RoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        residual = hidden_states
        
        # Sensory
        for i, layer in enumerate(self.sensory_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if i % 2 == 1:
                # Conv layer
                hidden_states = self.token2d(hidden_states, mode="seq_to_2d")
                hidden_states = layer(hidden_states)
                hidden_states = self.token2d(hidden_states, mode="d2_to_seq")
            else:
                # Attention layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=combined_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values, 
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = hidden_states + self.sensory_to_associative(F.silu(residual))
        
        # Associative
        for i, layer in enumerate(self.associative_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            if i % 2 == 1:
                hidden_states = self.token2d(hidden_states, mode="seq_to_2d")
                hidden_states = layer(hidden_states)
                hidden_states = self.token2d(hidden_states, mode="d2_to_seq")
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=combined_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    
        hidden_states = hidden_states + self.sensory_to_motor(F.silu(-residual))
        
        # Motor
        for i, layer in enumerate(self.motor_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=combined_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
class NeuroBLASTPreTrainedModel(PreTrainedModel):
    config_class = NeuroBLASTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NeuroBLASTBlock"]
    _skip_keys_device_placement = "past_key_values"

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

class NeuroBLASTForCausalLM(NeuroBLASTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = NeuroBLASTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

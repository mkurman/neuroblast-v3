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

import math
from typing import Optional, Tuple, Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import dot_product_attention
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.utils import logging

from .configuration_neuroblast_jax import NeuroBLASTConfig

logger = logging.get_logger(__name__)


class NeuroBLASTRMSNorm(nn.Module):
    hidden_size: int = 0
    eps: float = 1e-6
    dtype: Any = jnp.float32

    def setup(self):
        self.weight = self.param(
            "weight",
            lambda rng, shape: jnp.ones(shape, dtype=self.dtype),
            (self.hidden_size,),
        )

    def __call__(self, hidden_states):
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class NeuroBLASTMLP(nn.Module):
    config: Optional[NeuroBLASTConfig] = None
    dtype: Any = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)
        
        # Activation function
        if self.config.hidden_act == "silu":
            self.act_fn = nn.silu
        elif self.config.hidden_act == "gelu":
            self.act_fn = nn.gelu
        elif self.config.hidden_act == "relu":
            self.act_fn = nn.relu
        else:
            raise ValueError(f"Unsupported activation: {self.config.hidden_act}")

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NeuroBLASTAttention(nn.Module):
    config: Optional[NeuroBLASTConfig] = None
    layer_idx: int = 0
    use_rope: bool = True
    dtype: Any = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = getattr(
            self.config, "head_dim", self.hidden_size // self.num_heads
        )
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_output_dim = self.num_heads * self.head_dim
        
        self.q_proj = nn.Dense(
            self.attn_output_dim, 
            use_bias=self.config.attention_bias, 
            dtype=self.dtype
        )
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, 
            use_bias=self.config.attention_bias, 
            dtype=self.dtype
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, 
            use_bias=self.config.attention_bias, 
            dtype=self.dtype
        )
        self.o_proj = nn.Dense(
            self.hidden_size, 
            use_bias=self.config.attention_bias, 
            dtype=self.dtype
        )
        
        self.q_norm = NeuroBLASTRMSNorm(self.head_dim, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.k_norm = NeuroBLASTRMSNorm(self.head_dim, eps=self.config.rms_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        deterministic: bool = True,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # RoPE
        if self.use_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
        if self.num_key_value_groups > 1:
            key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
            value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)
            
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum("...qd,...kd->...qk", query_states, key_states) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.softmax(attn_weights, axis=-1)
        
        if self.config.attention_dropout > 0.0 and not deterministic:
             attn_weights = nn.dropout(
                 nn.make_rng(), 
                 attn_weights, 
                 deterministic=deterministic, 
                 rate=self.config.attention_dropout
             )
             
        attn_output = jnp.einsum("...qk,...kd->...qd", attn_weights, value_states)
        
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.attn_output_dim)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output



class NeuroBLASTRMSNorm2d(nn.Module):
    dim: int = 0
    eps: float = 1e-6
    dtype: Any = jnp.float32

    def setup(self):
        self.weight = self.param(
            "weight",
            lambda rng, shape: jnp.ones(shape, dtype=self.dtype),
            (self.dim,),
        )

    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(variance + self.eps)
        return self.weight * x_norm


class NeuroBLASTCausalConv2DBlock(nn.Module):
    config: Optional[NeuroBLASTConfig] = None
    dilation: int = 1
    layer_idx: int = 0
    dtype: Any = jnp.float32

    def setup(self):
        k = self.config.kernel_size
        d = self.config.hidden_size
        s = self.config.scale
        
        if s == 1:
            self.conv = nn.Conv(
                features=d,
                kernel_size=(k, k),
                kernel_dilation=(1, self.dilation),
                padding=(k // 2, 0),
                use_bias=False, 
                dtype=self.dtype,
            )
            self.use_gating = False
            self.use_projection = False
        elif s > 1:
            internal_dim = int(d * s)
            self.conv = nn.Conv(
                features=internal_dim,
                kernel_size=(k, k),
                kernel_dilation=(1, self.dilation),
                padding=(k // 2, 0),
                use_bias=False,
                dtype=self.dtype,
            )
            self.use_gating = True
            self.use_projection = False
        else:
            internal_dim = max(int(d * s), d // 4)
            self.conv = nn.Conv(
                features=internal_dim,
                kernel_size=(k, k),
                kernel_dilation=(1, self.dilation),
                padding=(k // 2, 0),
                use_bias=False,
                dtype=self.dtype,
            )
            self.use_gating = False
            self.use_projection = True
            self.proj_back = nn.Conv(features=d, kernel_size=(1, 1), use_bias=False, dtype=self.dtype)

        self.norm_in = NeuroBLASTRMSNorm2d(d, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.norm_out = NeuroBLASTRMSNorm2d(d, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout)

    def __call__(self, x, deterministic: bool = True):
        B, H, W, C = x.shape  # Store original W for cropping!
        residual = x
        y = self.norm_in(x)
        
        k = self.config.kernel_size
        pad_w = (k - 1) * self.dilation
        
        y_pad = jnp.pad(y, ((0, 0), (0, 0), (pad_w, 0), (0, 0)), mode='constant')
        
        y = self.conv(y_pad)
        
        y = y[:, :, -W:, :]
        
        if self.use_gating:
            gate, val = jnp.split(y, 2, axis=-1)
            y = val * nn.softmax(gate, axis=-1)
        elif self.use_projection:
            y = self.proj_back(y)
            
        y = self.norm_out(y)
        
        x = residual + self.dropout(y, deterministic=deterministic)
        return x


class NeuroBLASTDecoderLayer(nn.Module):
    config: Optional[NeuroBLASTConfig] = None
    layer_idx: int = 0
    attention_type: str = "full_attention"
    dtype: Any = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        
        if self.attention_type == "linear_attention":
            self.self_attn = NeuroBLASTLinearAttention(
                config=self.config,
                layer_idx=self.layer_idx,
                query_feature_map_name=self.config.query_feature_map,
                kv_feature_map_name=self.config.kv_feature_map,
                dtype=self.dtype,
            )
        else:
            self.self_attn = NeuroBLASTAttention(
                config=self.config,
                layer_idx=self.layer_idx,
                use_rope=(self.attention_type != "no_rope"),
                dtype=self.dtype,
            )
            
        self.mlp = NeuroBLASTMLP(self.config, dtype=self.dtype)
        self.input_layernorm = NeuroBLASTRMSNorm(self.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.post_attention_layernorm = NeuroBLASTRMSNorm(self.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        deterministic: bool = True,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            deterministic=deterministic,
        )
            
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class NeuroBLASTToken2D(nn.Module):
    dtype: Any = jnp.float32

    def __call__(self, x, mode="seq_to_2d"):
        if mode == "seq_to_2d":
            # x: (B, L, C) → (B, H, W, C) where H=1, W=L
            # PyTorch: x.view(B, L, 1, C).permute(0, 3, 2, 1) → (B, C, 1, L) 
            # In Flax channels-last: (B, H=1, W=L, C)
            B, L, C = x.shape
            x = x.reshape(B, 1, L, C)  # (B, H=1, W=L, C)
            return x
        else:
            # x: (B, H, W, C) → (B, L, C) where L = W*H
            # PyTorch: x.permute(0, 3, 2, 1).view(B, W * H, C)
            #   Permute (B,C,H,W) → (B,W,H,C), then flatten to (B,W*H,C)
            #   This means W varies first (causal ordering)
            # 
            # Flax: transpose (B, H, W, C) → (B, W, H, C), then flatten
            B, H, W, C = x.shape
            x = x.transpose(0, 2, 1, 3)  # (B, H, W, C) → (B, W, H, C) 
            x = x.reshape(B, W * H, C)   # (B, W, H, C) → (B, W*H, C)
            return x


class NeuroBLASTRotaryEmbedding(nn.Module):
    config: Optional[NeuroBLASTConfig] = None
    dtype: Any = jnp.float32

    def setup(self):
        self.dim = self.config.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings
        self.base = self.config.rope_theta
        
        # Precompute freqs
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.inv_freq = inv_freq

    def __call__(self, x, position_ids):
        # x: (B, L, H, D)
        # position_ids: (B, L) or (1, L)
        
        inv_freq_expanded = self.inv_freq[None, :, None] # (1, D/2, 1)
        
        # position_ids: (B, L)
        # We want (B, L, D/2)
        
        position_ids_expanded = position_ids[:, :, None] # (B, L, 1)
        
        # freqs: (B, L, D/2)
        freqs = jnp.matmul(position_ids_expanded.astype(jnp.float32), self.inv_freq[None, None, :])
        
        # emb: (B, L, D)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        
        # Expand for heads: (B, L, 1, D)
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
        
        return cos, sin


class NeuroBLASTModel(nn.Module):
    config: Optional[NeuroBLASTConfig] = None
    dtype: Any = jnp.float32

    def setup(self):
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        
        self.token2d = NeuroBLASTToken2D(dtype=self.dtype)
        
        sensory_layers = []
        dilatation_step = 1
        for i in range(self.config.num_sensory_layers):
            if i % 2 == 0:
                layer = NeuroBLASTDecoderLayer(
                    self.config, layer_idx=i, attention_type="full_attention", dtype=self.dtype, name=f"sensory_layers_{i}"
                )
            else:
                dilation = min(2 ** ((i - 1) // dilatation_step), 8)
                layer = NeuroBLASTCausalConv2DBlock(
                    self.config, dilation=dilation, layer_idx=i, dtype=self.dtype, name=f"sensory_layers_{i}"
                )
            sensory_layers.append(layer)
        self.sensory_layers = sensory_layers
            
        self.sensory_to_associative = NeuroBLASTRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        
        associative_layers = []
        next_layer_type = "full_attention"
        for i in range(self.config.num_associative_layers):
            idx = i + self.config.num_sensory_layers
            if i % 2 == 0:
                layer = NeuroBLASTDecoderLayer(
                    self.config, layer_idx=idx, attention_type=next_layer_type, dtype=self.dtype, name=f"associative_layers_{i}"
                )
                if next_layer_type == "full_attention":
                    next_layer_type = "no_rope"
                else:
                    next_layer_type = "full_attention"
            else:
                dilation = min(2 ** ((i - 1) // dilatation_step), 8)
                layer = NeuroBLASTCausalConv2DBlock(
                    self.config, dilation=dilation, layer_idx=idx, dtype=self.dtype, name=f"associative_layers_{i}"
                )
            associative_layers.append(layer)
        self.associative_layers = associative_layers
            
        self.sensory_to_motor = NeuroBLASTRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        
        motor_layers = []
        next_layer_type = "full_attention"
        for i in range(self.config.num_motor_layers):
            idx = i + self.config.num_sensory_layers + self.config.num_associative_layers
            layer = NeuroBLASTDecoderLayer(
                self.config, layer_idx=idx, attention_type=next_layer_type, dtype=self.dtype, name=f"motor_layers_{i}"
            )
            if next_layer_type == "full_attention":
                next_layer_type = "no_rope"
            else:
                next_layer_type = "full_attention"
            motor_layers.append(layer)
        self.motor_layers = motor_layers
            
        self.norm = NeuroBLASTRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)
        self.rotary_emb = NeuroBLASTRotaryEmbedding(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = jnp.arange(seq_len, dtype="i4")[None, :]
            
        # Create attention mask
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype="i4")

        # Build a boolean mask that enforces both padding and causality, then turn it
        # into an additive bias for the attention logits.
        attention_mask_bool = attention_mask.astype(bool)
        causal_mask = nn.make_causal_mask(attention_mask_bool)
        padding_mask = nn.make_attention_mask(attention_mask_bool, attention_mask_bool)
        combined_mask = nn.combine_masks(causal_mask, padding_mask)
        attention_bias = jnp.where(
            combined_mask,
            jnp.array(0.0, dtype=self.dtype),
            jnp.array(jnp.finfo(self.dtype).min, dtype=self.dtype),
        )
        
        # Embedding lookup
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Zero out padding token embeddings (equivalent to PyTorch's padding_idx)
        # attention_mask is (B, L) with 1 for real tokens, 0 for padding
        # Expand to (B, L, 1) to broadcast across hidden_size dimension
        embedding_mask = attention_mask[:, :, None].astype(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds * embedding_mask
        
        hidden_states = inputs_embeds
        
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        residual = hidden_states
        
        # Sensory
        for i, layer in enumerate(self.sensory_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            if i % 2 == 1:
                hidden_states = self.token2d(hidden_states, mode="seq_to_2d")
                hidden_states = layer(hidden_states, deterministic=deterministic)
                hidden_states = self.token2d(hidden_states, mode="d2_to_seq")
            else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_bias,
                        position_embeddings=position_embeddings,
                        deterministic=deterministic,
                    )
                
        hidden_states = hidden_states + self.sensory_to_associative(nn.silu(residual))
        
        # Associative
        for i, layer in enumerate(self.associative_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            if i % 2 == 1:
                hidden_states = self.token2d(hidden_states, mode="seq_to_2d")
                hidden_states = layer(hidden_states, deterministic=deterministic)
                hidden_states = self.token2d(hidden_states, mode="d2_to_seq")
            else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_bias,
                        position_embeddings=position_embeddings,
                        deterministic=deterministic,
                    )
                
        hidden_states = hidden_states + self.sensory_to_motor(nn.silu(-residual))
        
        # Motor
        for i, layer in enumerate(self.motor_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_bias,
                position_embeddings=position_embeddings,
                deterministic=deterministic,
            )
            
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
            
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class NeuroBLASTForCausalLMModule(nn.Module):
    config: Optional[NeuroBLASTConfig] = None
    dtype: Any = jnp.float32

    def setup(self):
        self.model = NeuroBLASTModel(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return (logits,) + outputs[1:]
            
        return FlaxCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class NeuroBLASTForCausalLM(FlaxPreTrainedModel):
    module_class = NeuroBLASTForCausalLMModule
    config_class = NeuroBLASTConfig

    def __init__(
        self,
        config: NeuroBLASTConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = NeuroBLASTForCausalLMModule(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unflatten_dict(random_params) | unflatten_dict(params))
            return FrozenDict(random_params)
        return random_params

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if params is None:
            params = self.params

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids),
            attention_mask=jnp.array(attention_mask) if attention_mask is not None else None,
            position_ids=jnp.array(position_ids) if position_ids is not None else None,
            deterministic=not train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
        )

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput



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

"""vLLM-compatible NeuroBLAST model implementation for vLLM 0.12.0+."""

from collections.abc import Iterable
from typing import List, Optional, Tuple, Union, Iterable, Dict, Any
from transformers import PretrainedConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

try:
    from vllm.utils.torch_utils import direct_register_custom_op
except ImportError:
    direct_register_custom_op = None

from vllm.forward_context import get_forward_context

# Global static cache
_GLOBAL_CONV_CACHE: Optional[torch.Tensor] = None

# Global layer registry for CustomOp to find layer instances
_LAYER_REGISTRY: Dict[str, Any] = {}

def get_conv_cache(
    num_conv_layers: int,
    cache_config: CacheConfig,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """Get or create the global static conv state cache."""
    global _GLOBAL_CONV_CACHE
    if _GLOBAL_CONV_CACHE is None:
        # Calculate total slots from cache config
        # num_gpu_blocks * block_size
        total_slots = cache_config.num_gpu_blocks * cache_config.block_size
        
        # Allocate global cache: (num_layers, total_slots, hidden_size)
        # We store 1 vector per slot. We reconstruct the window by gathering.
        # This is memory efficient (reuse KV cache memory pattern) and robust.
        _GLOBAL_CONV_CACHE = torch.zeros(
            num_conv_layers, total_slots, cache_config.hidden_size, # Assuming hidden_size is passed or standard
            dtype=dtype, device=device
        )
    return _GLOBAL_CONV_CACHE

def _get_past_indices(
    current_positions: torch.Tensor, # (B,)
    block_tables: torch.Tensor,      # (B, max_blocks)
    block_size: int,
    window_size: int
) -> torch.Tensor: # (B, window_size)
    """Compute physical slot indices for the past window_size tokens."""
    # We want [pos-window_size, ..., pos-1]
    # Shape logic:
    # positions: (B, 1)
    # offsets: (1, window_size) -> [-window_size, ..., -1]
    # past_pos: (B, window_size)
    
    B = current_positions.shape[0]
    offsets = torch.arange(-window_size, 0, device=current_positions.device).view(1, window_size)
    past_pos = current_positions.view(B, 1) + offsets # (B, window_size)
    
    # Clamp to 0 (though should not handle negative positions in valid generation)
    past_pos = torch.clamp(past_pos, min=0)
    
    # Compute block indices and offsets
    block_indices = past_pos // block_size
    block_offsets = past_pos % block_size
    
    # Gather physical block numbers
    # block_tables is (B, max_blocks)
    # We gather from dim 1 using block_indices
    # physical_blocks: (B, window_size)
    physical_blocks = torch.gather(block_tables, 1, block_indices.long())
    
    # Compute flattened physical slot indices
    # slot = physical_block * block_size + block_offset
    slot_indices = physical_blocks.long() * block_size + block_offsets.long()
    
    return slot_indices

class NeuroBLASTConfig(PretrainedConfig):
    model_type = "neuroblast"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        energy_dim=128,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_associative_layers=16,
        num_sensory_layers=8,
        num_motor_layers=8,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        attention_every=0,
        dropout=0.0,
        scale=1.0,
        kernel_size=5,
        temporal_kernel_size=5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.energy_dim = energy_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.num_associative_layers = num_associative_layers
        self.num_sensory_layers = num_sensory_layers

        self.num_motor_layers = num_motor_layers

        if (
            num_hidden_layers
            != num_associative_layers + num_sensory_layers + num_motor_layers
        ):
            self.num_hidden_layers = (
                num_associative_layers + num_sensory_layers + num_motor_layers
            )
            warnings.warn(
                f"num_hidden_layers ({num_hidden_layers}) is not equal to num_associative_layers ({num_associative_layers}) + num_sensory_layers ({num_sensory_layers}) + num_motor_layers ({num_motor_layers}). Setting num_hidden_layers to {num_associative_layers + num_sensory_layers + num_motor_layers}."
            )
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_every = attention_every
        self.scale = scale
        self.kernel_size = kernel_size
        self.temporal_kernel_size = temporal_kernel_size
        self.dropout = dropout

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                ("full_attention") for i in range(self.num_hidden_layers)
            ]
        self.num_hidden_layers = (
            num_associative_layers // 2 + num_sensory_layers // 2 + num_motor_layers
        )
        self.num_conv_layers = self.num_hidden_layers - num_motor_layers
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class NeuroBLASTRMSNorm2d(nn.Module):
    """RMSNorm for 2D convolution outputs (B, C, H, W)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.view(1, -1, 1, 1) * x_norm.to(input_dtype)

def neuroblast_conv2d(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Execute Conv2D forward outside CUDA graph context."""
    # forward_context_obj: ForwardContext = get_forward_context()
    # Get the layer instance from global registry using layer_name (prefix)
    if layer_name in _LAYER_REGISTRY:
        self = _LAYER_REGISTRY[layer_name]
        self.forward_cuda(hidden_states=hidden_states, positions=positions, output=output)
    else:
        # Should not happen
        pass


def neuroblast_conv2d_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for tracing - does nothing."""
    return


# Register the custom op - this makes it callable as torch.ops.vllm.neuroblast_conv2d
if direct_register_custom_op is not None:
    direct_register_custom_op(
        op_name="neuroblast_conv2d",
        op_func=neuroblast_conv2d,
        mutates_args=["output"],
        fake_impl=neuroblast_conv2d_fake,
    )

@CustomOp.register("neuroblast_conv2d")
class NeuroBLASTCausalConv2DBlock(CustomOp):
    """Conv2D block with CUDA graph compatible state caching."""
    
    def __init__(
        self, 
        vllm_config: VllmConfig,
        dilation: int = 1, 
        layer_idx: int = 0,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.cache_config = vllm_config.cache_config
        k = self.config.kernel_size
        self.k = k
        self.d = self.config.hidden_size
        self.dilation = dilation
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.cache_config = self.cache_config if not cache_config else cache_config
        
        scale = self.config.scale
        hidden_size = self.config.hidden_size
        
        self.conv_padding = (k // 2, 0)

        if scale == 1:
            self.conv = Conv2dLayer(
                hidden_size,
                hidden_size,
                kernel_size=(k, k),
                stride=(1, 1),
                dilation=(1, dilation),
                padding=self.conv_padding,
                bias=False,
            )
            self.use_gating = False
            self.use_projection = False
        elif scale > 1:
            internal_dim = int(hidden_size * scale)
            self.conv = Conv2dLayer(
                hidden_size,
                internal_dim,
                kernel_size=(k, k),
                stride=(1, 1),
                dilation=(1, dilation),
                padding=self.conv_padding,
                bias=False,
            )
            self.use_gating = True
            self.use_projection = False
        else:
            internal_dim = max(int(hidden_size * scale), hidden_size // 4)
            self.conv = Conv2dLayer(
                hidden_size,
                internal_dim,
                kernel_size=(k, k),
                stride=(1, 1),
                dilation=(1, dilation),
                padding=self.conv_padding,
                bias=False,
            )
            self.use_gating = False
            self.use_projection = True
            self.proj_back = nn.Conv2D(
                internal_dim, hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False
            )
        
        self.norm_in = NeuroBLASTRMSNorm2d(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.norm_out = NeuroBLASTRMSNorm2d(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # Register in global registry for CustomOp
        if prefix:
            _LAYER_REGISTRY[prefix] = self


    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        torch.ops.vllm.neuroblast_conv2d(x, positions, output, self.prefix)
        return output



    def forward_native(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        self.forward_cuda(hidden_states, positions, output)

    def forward_cuda(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        x = hidden_states
        B, C, H, W = x.shape
        
        # 2. Extract metadata from ForwardContext (vLLM mechanism)
        slot_mapping = None
        block_tables = None
        
        try:
            forward_context_obj = get_forward_context()
            if forward_context_obj.attn_metadata is not None:
                meta_dict = forward_context_obj.attn_metadata
                
                # Handle potential list wrapper (e.g. for microbatches)
                if isinstance(meta_dict, list):
                    meta_dict = meta_dict[0]
                
                if self.prefix in meta_dict:
                    layer_meta = meta_dict[self.prefix]
                else:
                    # Fallback: Borrow metadata from the first available Attention layer
                    if meta_dict:
                        first_key = next(iter(meta_dict))
                        layer_meta = meta_dict[first_key]
                    else:
                        layer_meta = None

                if layer_meta is not None:
                    slot_mapping = getattr(layer_meta, "slot_mapping", None)
                    block_tables = getattr(layer_meta, "block_table", None)
        except Exception as e:
            # print(f"DEBUG EXCEPTION in forward_cuda: {e}")
            pass

        # DEBUG PRINTS
        # if slot_mapping is not None and W == 1:
        #      print(f"L{self.layer_idx} DEBUG: W={W}, slot_mapping found")
        
        if slot_mapping is None:
             pass # Will fall through to else cases
             
        # Initialize/Get global cache
        if self.cache_config is None:
             try:
                 # Attempt restart logic or find config if safe? No.
                 pass
             except:
                 pass
                 
        global _GLOBAL_CONV_CACHE
        
        # Determine if we need to allocate or re-allocate the cache
        need_allocation = False
        target_slots = None

        # print(self.vllm_config)
        # print(self.cache_config)
        
        if self.cache_config and self.cache_config.block_size is not None:
            # We have valid block info from vLLM
            num_blocks = self.cache_config.num_gpu_blocks if self.cache_config.num_gpu_blocks is not None else 1
            block_size = self.cache_config.block_size
            target_slots = self.vllm_config.model_config.max_model_len if self.vllm_config.model_config.max_model_len is not None else self.config.max_position_embeddings
            
            if _GLOBAL_CONV_CACHE is None:
                need_allocation = True
            elif _GLOBAL_CONV_CACHE.shape[1] != target_slots:
                # RE-ALLOCATION: Cache exists but size mismatch (e.g., fallback from profiling)
                # print(f"DEBUG: Re-allocating Cache from {_GLOBAL_CONV_CACHE.shape[1]} to {target_slots} slots")
                need_allocation = True
        else:
            # PROFILING / FALLBACK MODE
            # num_gpu_blocks is None during profiling.
            # Use a very large fallback to handle any slot vLLM might assign.
            # This will be re-allocated after profiling when num_gpu_blocks is known.
            if _GLOBAL_CONV_CACHE is None:
                fallback_blocks = 16
                print(f"DEBUG: Using fallback blocks: {fallback_blocks}")
                block_size = 16
                target_slots = self.vllm_config.model_config.max_model_len if self.vllm_config.model_config.max_model_len is not None else self.config.max_position_embeddings
                need_allocation = True
        
        if need_allocation and target_slots is not None:
            # print(f"DEBUG: Allocating Conv Cache: {self.config.num_hidden_layers} x {target_slots} x {C}")
            _GLOBAL_CONV_CACHE = torch.zeros(
                self.config.num_conv_layers, target_slots, C,
                dtype=x.dtype, device=x.device
            )
        
        conv_cache = _GLOBAL_CONV_CACHE
        y_in = self.norm_in(hidden_states) # (B, C, 1, W) assuming flattened batch
        
        if conv_cache is not None and slot_mapping is not None:
             # STATEFUL EXECUTION
             y_flat = y_in.permute(0, 3, 2, 1).reshape(-1, C) # (B*W, C)
             
             conv_layer_idx = self.layer_idx // 2
             
             # Identify trash slot index
             trash_slot = conv_cache.shape[1] - 1
             
             if W > 1:
                 # PREFILL / PROMPT PHASE
                 pad_w = (self.k - 1) * self.dilation
                 y_pad = F.pad(y_in, (pad_w, 0, 0, 0)) # (B, C, 1, W + pad)
                 # PREFILL PHASE (W > 1)
                 # Update cache with last window_size tokens of the INPUT hidden_states
                 window_size = (self.k - 1) * self.dilation
                 
                 # Determine valid length from slot_mapping
                 if slot_mapping.dim() == 2:
                     valid_len = slot_mapping.shape[1]
                 else:
                     valid_len = slot_mapping.shape[0]
                 
                 # hidden_states might be padded (right). Extract valid prefix.
                 # Assuming padding is on the right and start aligns.
                 unpadded_x = hidden_states[:, :, :, :valid_len]
                 
                 seq_len = unpadded_x.shape[-1]
                 num_to_cache = min(seq_len, window_size)
                 
                 # Extract last num_to_cache inputs
                 x_update = unpadded_x[:, :, :, -num_to_cache:] # (B, C, 1, num)
                 
                 # Reshape to (B*num, C) for index_copy_
                 x_flat = x_update.permute(0, 3, 2, 1).flatten(0, 2).squeeze(1)
                 
                 # Get corresponding slots
                 # slot_mapping is (B, W) or (num_tokens). Slice last num_to_cache columns.
                 if slot_mapping.dim() == 2:
                     current_slots = slot_mapping[:, -num_to_cache:].flatten()
                 else:
                     # 1D case
                     current_slots = slot_mapping[-num_to_cache:]
                 
                 # Mask out invalid slots using Trash Slot redirect
                 # Handle BOTH negative indices AND indices >= cache size
                 layer_cache = conv_cache[conv_layer_idx]
                 safe_slots = current_slots.clone()
                 cache_size = layer_cache.shape[0]
                 invalid_mask = (current_slots < 0) | (current_slots >= cache_size)
                 safe_slots[invalid_mask] = trash_slot
                 
                 layer_cache.index_copy_(0, safe_slots, x_flat)
                 
                 y = self.conv(y_pad)
                 y = y[:, :, :, -W:].contiguous()
                 
             else:
                 # DECODE PHASE (W=1)
                 if self.cache_config:
                      block_size = self.cache_config.block_size
                 else:
                      block_size = 16 # Fallback default
                 
                 window_size = (self.k - 1) * self.dilation
                 
                 past_indices = _get_past_indices(
                     positions.flatten(), block_tables, block_size, window_size
                 ) # (B, k-1)
                 
                 # Decode Read Safety - handle BOTH negative AND out-of-bounds indices
                 layer_cache = conv_cache[conv_layer_idx] # (total_slots, C)
                 cache_size = layer_cache.shape[0]
                 safe_past_indices = past_indices.long()
                 invalid_mask = (safe_past_indices < 0) | (safe_past_indices >= cache_size)
                 safe_past_indices[invalid_mask] = trash_slot
                 
                 past_states = layer_cache[safe_past_indices.flatten()] # (B*window_size, C)
                 past_states = past_states.view(B, window_size, C).permute(0, 2, 1).unsqueeze(2) # (B, C, 1, window_size)
                 
                 current_state = y_flat.unsqueeze(2).unsqueeze(3) # (B, C, 1, 1)
                 # y_flat is (B, C). 
                 # We need (B, C, 1, 1)
                 current_state_4d = y_in 
                 
                 # Concatenate: past (B, C, 1, k-1) + current (B, C, 1, 1)
                 x_conv = torch.cat([past_states, current_state_4d], dim=3) # (B, C, 1, k)
                 
                 y = self.conv(x_conv) # (B, C, 1, 1)
                 
                 # Update Cache Safety - handle BOTH negative AND out-of-bounds indices
                 current_slots = slot_mapping.flatten()
                 safe_slots = current_slots.clone()
                 invalid_mask = (current_slots < 0) | (current_slots >= cache_size)
                 safe_slots[invalid_mask] = trash_slot
                 
                 layer_cache.index_copy_(0, safe_slots, y_flat)

        else:
             # FALLBACK (Stateless)
             # if self.cache_config and slot_mapping is None:
             #    print(f"L{self.layer_idx} FALLBACK: Stateless (No slot_mapping or No Cache)")
                 
             pad_w = (self.k - 1) * self.dilation
             y_pad = F.pad(y_in, (pad_w, 0, 0, 0))
             y = self.conv(y_pad)
             y = y[:, :, :, -W:].contiguous()
        
        # Final Processing
        if self.use_gating:
            gate, val = torch.chunk(y, 2, dim=1)
            y = val * F.softmax(gate, dim=1)
        elif self.use_projection:
             y = self.proj_back(y)
        
        y = self.norm_out(y)
        output.copy_(x + y)



class NeuroBLASTMLP(nn.Module):
    """MLP with SwiGLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class NeuroBLASTAttention(nn.Module):
    """Multi-head attention with optional RoPE."""

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        use_rope: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_rope = use_rope

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters={"rope_theta": config.rope_theta, "rope_type": "default"},
            )
        else:
            self.rotary_emb = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply QK norm
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(-1, self.q_size)
        k = k.view(-1, self.kv_size)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NeuroBLASTDecoderLayer(nn.Module):
    """Decoder layer with attention and MLP."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_rope: bool = True,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = NeuroBLASTAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=max_position_embeddings,
            use_rope=use_rope,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = NeuroBLASTMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states

        return hidden_states


class NeuroBLASTToken2D(nn.Module):
    """Converts between sequence and 2D tensor formats."""
    def forward(self, x: torch.Tensor, mode: str = "seq_to_2d") -> torch.Tensor:
        if mode == "seq_to_2d":
            # vLLM input: (seq_len, hidden_size) -> (1, C, 1, seq_len)
            L, C = x.shape
            x = x.unsqueeze(0)  # (1, L, C)
            x = x.view(1, L, 1, C)  # (1, L, 1, C)
            x = x.permute(0, 3, 2, 1).contiguous()  # (1, C, 1, L) = (B, C, H, W)
            return x
        else:
            # d2_to_seq: (1, C, 1, L) -> (L, C)
            B, C, H, W = x.shape
            x = x.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C) = (1, L, 1, C)
            x = x.view(B, W * H, C).contiguous()  # (1, L, C)
            x = x.squeeze(0).contiguous()  # (L, C)
            return x


class NeuroBLASTModel(nn.Module):
    """NeuroBLAST model with three-stage cortical architecture."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )

        self.token2d = NeuroBLASTToken2D()

        # Sensory layers
        self.sensory_layers = nn.ModuleList()
        self.sensory_attn_indices = []
        dilatation_step = 1
        for i in range(config.num_sensory_layers):
            if i % 2 == 0:
                layer = NeuroBLASTDecoderLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.sensory_layers.{i}",
                    use_rope=True,
                )
                self.sensory_attn_indices.append(i)
            else:
                dilation = min(2 ** ((i - 1) // dilatation_step), 8)
                layer = NeuroBLASTCausalConv2DBlock(
                    vllm_config=vllm_config, 
                    dilation=dilation, 
                    layer_idx=i,
                    cache_config=cache_config,
                    prefix=f"{prefix}.sensory_layers.{i}")
            self.sensory_layers.append(layer)

        self.sensory_to_associative = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Associative layers
        self.associative_layers = nn.ModuleList()
        self.associative_attn_indices = []
        next_use_rope = True
        for i in range(config.num_associative_layers):
            if i % 2 == 0:
                layer = NeuroBLASTDecoderLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.associative_layers.{i}",
                    use_rope=next_use_rope,
                )
                self.associative_attn_indices.append(i)
                next_use_rope = not next_use_rope
            else:
                dilation = min(2 ** ((i - 1) // dilatation_step), 8)
                layer = NeuroBLASTCausalConv2DBlock(
                    vllm_config=vllm_config, 
                    dilation=dilation, 
                    layer_idx=i,
                    cache_config=cache_config,
                    prefix=f"{prefix}.associative_layers.{i}")
            self.associative_layers.append(layer)

        self.sensory_to_motor = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Motor layers
        self.motor_layers = nn.ModuleList()
        next_use_rope = True
        for i in range(config.num_motor_layers):
            layer = NeuroBLASTDecoderLayer(
                vllm_config=vllm_config,
                prefix=f"{prefix}.motor_layers.{i}",
                use_rope=next_use_rope,
            )
            next_use_rope = not next_use_rope
            self.motor_layers.append(layer)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        residual = hidden_states
        # Sensory cortex
        for i, layer in enumerate(self.sensory_layers):
            if i % 2 == 1:
                # Conv layer - flatten residual first
                hidden_states = self.token2d(hidden_states, mode="seq_to_2d")
                hidden_states = layer(hidden_states, positions=positions)
                hidden_states = self.token2d(hidden_states, mode="d2_to_seq")
            else:
                # Attention layer
                hidden_states = layer(positions, hidden_states)

        hidden_states = hidden_states + self.sensory_to_associative(F.silu(residual))

        # Associative cortex
        for i, layer in enumerate(self.associative_layers):
            if i % 2 == 1:
                hidden_states = self.token2d(hidden_states, mode="seq_to_2d")
                hidden_states = layer(hidden_states, positions=positions)
                hidden_states = self.token2d(hidden_states, mode="d2_to_seq")
            else:
                hidden_states = layer(positions, hidden_states)

        hidden_states = hidden_states + self.sensory_to_motor(F.silu(-residual))

        # Motor cortex
        for layer in self.motor_layers:
            hidden_states = layer(positions, hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class NeuroBLASTForCausalLM(nn.Module):
    """NeuroBLAST model for causal language modeling."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = NeuroBLASTModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}model" if prefix else "model",
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}lm_head" if prefix else "lm_head",
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        
        return loaded_params

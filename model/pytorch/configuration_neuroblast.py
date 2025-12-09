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

"""NeuroBLASTConfig model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

import math
import warnings

logger = logging.get_logger(__name__)


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
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                ("full_attention") for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["NeuroBLASTConfig"]

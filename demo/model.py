# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import math
import time
from prettytable import PrettyTable
import warnings
import traceback
from typing import List, Optional, Tuple, Union
import socket
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaMLP,
    LlamaRMSNorm,
    _CONFIG_FOR_DOC,
    logger,
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
    LlamaDecoderLayer,
    LlamaAttention,
)
from pl_lib import (
    PLAB,
    recv_tensor,
    send_tensor,
    END_OF_GENERATE,
    CommProfiler,
    AttnProfiler,
)
import logging


class PLLlamaConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        rcd=128,
        rdc=128,
        cloud_ip="127.0.0.1",
        cloud_port=12345,
        layers_to_transform=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            **kwargs,
        )
        self.rcd = rcd
        self.rdc = rdc
        self.cloud_ip = cloud_ip
        self.cloud_port = cloud_port
        self.layers_to_transform = layers_to_transform


class PLLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PLLlamaConfig, add_lora: bool):
        LlamaAttention.__init__(self, config)
        self.add_lora = add_lora
        if add_lora:
            self.lora_AB = PLAB(
                config.hidden_size, config.rcd, config.rdc, config.hidden_size
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        s: socket.socket = None,
        buffer_size=int(2048e3),
        profiler: AttnProfiler = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:

            def do_lora(hidden_states, s, buffer_size):
                return self.lora_AB(hidden_states, s, buffer_size)

            def do_qkv(hidden_states):
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                return query_states, key_states, value_states

            if self.add_lora:
                # # attempts of parallism with concurrent.futures, actually slower
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     qkv_thread = executor.submit(do_qkv, hidden_states)
                #     lora_thread = executor.submit(do_lora, hidden_states, s, buffer_size)
                #     query_states, key_states, value_states = qkv_thread.result()
                #     xq, xk, xv = lora_thread.result()
                if profiler is not None:
                    start = time.time()
                query_states, key_states, value_states = do_qkv(hidden_states)
                if profiler is not None:
                    qkv = time.time()
                    profiler.log_qkv(qkv - start)
                xq, xk, xv = do_lora(hidden_states, s, buffer_size)
                if profiler is not None:
                    lora = time.time()
                    profiler.log_lora(lora - qkv)
                query_states += xq.to(query_states.device)
                key_states += xk.to(query_states.device)
                value_states += xv.to(query_states.device)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PLLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: PLLlamaConfig, add_lora: bool):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.self_attn = PLLlamaAttention(config=config, add_lora=add_lora)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        s: socket.socket = None,
        buffer_size=int(2048e3),
        profiler=None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
            s=s,
            buffer_size=buffer_size,
            profiler=profiler,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
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


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class PLLlamaPreTrainedModel(PreTrainedModel):
    config_class = PLLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PLLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

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


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(PLLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: PLLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if config.layers_to_transform is not None:
            self.layers = nn.ModuleList(
                [
                    PLLlamaDecoderLayer(config, i in config.layers_to_transform)
                    for i in range(config.num_hidden_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    PLLlamaDecoderLayer(config, True)
                    for i in range(config.num_hidden_layers)
                ]
            )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        return_dict: Optional[bool] = None,
        s: socket.socket = None,
        profiler=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    s=s,
                    profiler=profiler,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCloud(PLLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PLLlamaConfig):
        super().__init__(config)

        self.model = LlamaModel(config)
        # Initialize weights and apply final processing
        self.post_init()

    def to_split_device(self):
        for n, m in self.named_modules():
            if "lora" in n:
                m = m.to("cpu").to(torch.float32)

    def cloud_to_fp16(self):
        for n, p in self.named_parameters():
            # if "lora" not in n:
            p = p.to(torch.float16)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        s: socket.socket = None,
        profiler=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if s is None:
            print("no socket is provided")
            return

        if inputs_embeds is None:
            inputs_embeds = recv_tensor(
                s, buffer_size=int(4096e6)
            )  # initial receive from device
        inputs_embeds = inputs_embeds.to(self.model.device).to(
            self.model.layers[0].self_attn.q_proj.weight.dtype
        )
        logging.debug(f"receiving word embedding of shape {inputs_embeds.shape}")

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
            return_dict=return_dict,
            s=s,
            profiler=profiler,
        )
        hidden_states = outputs[0]
        send_tensor(s, hidden_states[:, -1, :].unsqueeze(1))  # send last
        logging.debug(f"kv cache shape{outputs.past_key_values[0][0].shape}")
        s.recv(1024)  # prevent 2 consecutive send
        send_tensor(
            s, torch.tensor(outputs.past_key_values[0][0].shape, dtype=torch.int16)
        )  # send kv shape to device for mocking
        return outputs.past_key_values

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def my_generate(self, *args, **kwargs):
        """custom generate for cloud model

        adds loop for iterative sampling
        """
        s = kwargs.get("s", None)

        if s is None:
            print("no socket")
            return
        past_key_values = None
        profiler = AttnProfiler()
        while True:
            try:
                # For cloud model, each loop starts with receiving word embedding
                inputs_embeds = recv_tensor(s)

                past_key_values = self(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    s=s,
                    profiler=profiler,
                )
            except Exception:
                print("==========generate done===========")
                profiler.get_report()
                # traceback.print_exc()
                return


class LlamaForDevice(PLLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PLLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        from pl_lib import PLMStack

        if config.layers_to_transform is None:
            num_layers = config.num_hidden_layers
        else:
            num_layers = len(config.layers_to_transform)

        self.lora_M_stack = PLMStack(num_layers, config.rcd, config.rdc)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.new_gens = []
        self.tokenizer = None
        # Initialize weights and apply final processing
        self.post_init()

    def set_tokenizer(self, tokenizer):
        """set tokenizer for stream

        Args:
            tokenizer (_type_): _description_
        """
        self.tokenizer = tokenizer

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        s: socket.socket = None,
        comm_profiler: CommProfiler = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        if s is None:
            print("no socket provided to s")
            return

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds = self.embed_tokens(input_ids)
        send_tensor(s, inputs_embeds, profiler=comm_profiler)  # initial communication

        self.lora_M_stack(s, profiler=comm_profiler)

        hidden_states = recv_tensor(s, buffer_size=int(4096e5))  # final comm
        if self.prefill_end is None:
            # time it for throughput analysis
            self.prefill_end = time.time()
        s.sendall("hi".encode())
        past_kv_shape = recv_tensor(s, buffer_size=1024)
        # hidden_states = outputs[0] # og
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        # stream output
        if self.tokenizer is not None:
            token_id = torch.argmax(logits, -1).item()
            old_full = self.tokenizer.decode(self.new_gens)
            self.new_gens.append(token_id)
            new_full = self.tokenizer.decode(self.new_gens)
            print(new_full[len(old_full) :], end="", flush=True)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        bs, h, seq, hi = past_kv_shape.tolist()
        past_key_values = [[torch.zeros((int(bs), int(h), int(seq + 1), int(hi)))]]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,  # mock on edge side
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        s = kwargs.get("s", None)
        comm_profiler = kwargs.get("comm_profiler", None)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "s": s,
                "comm_profiler": comm_profiler,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def my_generate(self, *args, **kwargs):
        """simple wrapper to tell cloud when a new query comes in"""
        s = kwargs.get("s", None)
        profiler = kwargs.get("comm_profiler", None)
        speed_profile = kwargs.pop("speed_profile", False)
        if s is None:
            print("no socket")
            return
        s.sendall("new".encode())
        s.recv(1024)
        with torch.no_grad():
            if speed_profile:
                self.prefill_end = None
                self.prefill_start = time.time()
            self.new_gens = []
            outs = self.generate(*args, **kwargs)
            if speed_profile:
                self.decode_end = time.time()
        s.sendall("finish".encode())

        if profiler is not None:
            profiler.get_report()

        if speed_profile:
            self.decode_time = self.decode_end - self.prefill_end
            self.prefill_time = self.prefill_end - self.prefill_start
            self.decode_tokens = outs.shape[1] - kwargs.get("input_ids").shape[1]
            self.prefill_tokens = kwargs.get("input_ids").shape[1]
            print("Throughput Stats")

            table = PrettyTable()

            # 设置列名
            table.field_names = ["Stages", "Tokens", "Time", "TPS"]

            table.align["Stages"] = "l"  # "l" 对应左对齐
            table.align["Tokens"] = "r"  # "r" 对应右对齐
            table.align["Time"] = "r"  # "r" 对应右对齐
            table.align["TPS"] = "r"  # "r" 对应右对齐
            table.add_row(
                [
                    "prefill",
                    self.prefill_tokens,
                    round(self.prefill_time, 2),
                    round(self.prefill_tokens / self.prefill_time, 2),
                ]
            )
            table.add_row(
                [
                    "decode",
                    self.decode_tokens,
                    round(self.decode_time, 2),
                    round(self.decode_tokens / self.decode_time, 2),
                ]
            )
            print(table)

        return outs

    def print_param_count(self):
        m_cnt = 0
        lmh = 0
        emb = 0
        for n, p in self.named_parameters():
            if "lora" in n:
                m_cnt += p.numel()
            elif "lm_head" in n:
                lmh += p.numel()
            elif "emb" in n:
                emb += p.numel()
        total = m_cnt + lmh + emb

        table = PrettyTable()

        # 设置列名
        table.field_names = ["Modules", "Param #", "Param %"]

        # 设置每列的对齐方式
        table.align["Modules"] = "l"  # "l" 对应左对齐
        table.align["Param #"] = "r"  # "r" 对应右对齐
        table.align["Param %"] = "r"  # "r" 对应右对齐
        table.add_row(["word emb", emb, round(emb / total * 100, 2)])
        table.add_row(["PrivateLoRA M", m_cnt, round(m_cnt / total * 100, 2)])
        table.add_row(["lm head", lmh, round(lmh / total * 100, 2)])
        print("Param statistics")
        print(table)

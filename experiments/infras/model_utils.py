from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Optional
from dataclasses import dataclass, field
import torch

DEFAULT_MODEL_INIT_KWARGS = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}


@dataclass
class BaseModelTokenizerHandler:
    model_cls = AutoModelForCausalLM
    model_name: str
    model_paths: Dict[str, str]
    model_from_pretrained_kwargs: Dict[str, str]
    tokenizer_from_pretrained_kwargs: Dict[str, str]
    tokenizer_paths: Optional[Dict[str, str]] = field(default=None)

    def model_post_init(self, model):
        print(model)
        return model

    def tokenizer_post_init(self, tokenizer):
        print(tokenizer)
        return tokenizer

    @classmethod
    def get_base_model(cls, config: str, sd=None, **kwargs):
        if config not in cls.model_paths.keys():
            raise ValueError()
        model = cls.model_cls.from_pretrained(
            cls.model_paths[config],
            **cls.model_from_pretrained_kwargs[config],
            state_dict=sd,
            **kwargs,
        )

        model = cls.model_post_init(cls, model)

        return model

    @classmethod
    def get_config(cls, config: str, **kwargs):
        if config not in cls.model_paths.keys():
            raise ValueError()
        return AutoConfig.from_pretrained(
            cls.model_paths[config], trust_remote_code=True
        )

    @classmethod
    def get_base_tokenizer(cls, config: str, **kwargs):
        if config not in cls.model_paths.keys():
            raise ValueError()
        tokenizer = AutoTokenizer.from_pretrained(
            cls.tokenizer_paths[config]
            if cls.tokenizer_paths
            else cls.model_paths[config],
            **cls.tokenizer_from_pretrained_kwargs[config],
        )

        tokenizer = cls.tokenizer_post_init(cls, tokenizer)
        return tokenizer

    @classmethod
    def get_base_model_and_tokenizer(cls, config, sd=None, **kwargs):
        return cls.get_base_model(config, sd, **kwargs), cls.get_base_tokenizer(
            config, **kwargs
        )


@dataclass
class Llama2BaseModelTokenizerHandler(BaseModelTokenizerHandler):
    model_paths = {
        "7bchat": "/mntnlp/common_base_model/llama2-7b-chat",
        "7b": "/mntnlp/common_base_model/llama2-7b",
        "13bchat": "/mntnlp/common_base_model/llama2-13b-chat",
        "13b": "/mntnlp/common_base_model/llama2-13b",
        "70bchat": "/mntnlp/common_base_model/llama2-70b-chat",
        "70b": "/mntnlp/common_base_model/llama2-70b",
        "30b": "/mntnlp/common_base_model/llama_30b",
    }
    model_from_pretrained_kwargs = {
        "7b": DEFAULT_MODEL_INIT_KWARGS,
        "7bchat": DEFAULT_MODEL_INIT_KWARGS,
        "13b": DEFAULT_MODEL_INIT_KWARGS,
        "70b": DEFAULT_MODEL_INIT_KWARGS,
        "70bchat": DEFAULT_MODEL_INIT_KWARGS,
        "30b": DEFAULT_MODEL_INIT_KWARGS,
        "13bchat": DEFAULT_MODEL_INIT_KWARGS,
    }
    tokenizer_from_pretrained_kwargs = {
        "7b": {"use_fast": False},
        "7bchat": {"use_fast": False},
        "13b": {"use_fast": False},
        "13bchat": {"use_fast": False},
        "70b": {"use_fast": False},
        "30b": {"use_fast": False},
        "70bchat": {"use_fast": False},
    }

    def tokenizer_post_init(self, tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return super().tokenizer_post_init(self, tokenizer)

    from transformers import AutoModelForCausalLM

    model_cls = AutoModelForCausalLM


@dataclass
class PrivateLlama(Llama2BaseModelTokenizerHandler):
    def model_post_init(self, model):
        # only lora M is trainable
        for n, p in model.named_parameters():
            if "lora_mobile" not in n:
                p.requires_grad = False
        return super().model_post_init(self, model)

    @classmethod
    def get_base_model(cls, config: str, sd=None, **kwargs):
        if config not in cls.model_paths.keys():
            raise ValueError()
        from mymodels.modeling_llama_pl import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            cls.model_paths[config],
            **cls.model_from_pretrained_kwargs[config],
            state_dict=sd,
            **kwargs,
        )
        model = cls.model_post_init(cls, model)

        return model


def get_base_model_and_tokenizer(model_name: str, weight_path: str = None, **kwargs):
    """common model and tokenizer getter, so we don't have tweak in training scripts

    Args:
        model_name (str): <model_name>-<scale/config>, something like llama2-7b, plllama-7b
        weight_path (str, optional): custom weight path. Defaults to None.

    Returns:
        model, tokenizer
    Examples:
        ```python
        model, tokenizer = get_base_model_and_tokenizer("llama2-7b")
        ```
    """
    name, config = model_name.split("-")
    assert (
        name in MODEL_NAME_HANDLER_MAP.keys()
    ), f"model {name} not registered, see `model_utils.py`"
    sd = None
    if weight_path is not None:
        sd = torch.load(weight_path)
    model, tokenizer = MODEL_NAME_HANDLER_MAP[name].get_base_model_and_tokenizer(
        config, sd, **kwargs
    )
    return model, tokenizer


def get_model_config(model_name: str, **kwargs):
    """model config getter

    Args:
        model_name (str): <model_name>-<scale/config>, something like llama2-7b, plllama-7b

    Returns:
        model_config
    """
    name, config = model_name.split("-")
    assert name in MODEL_NAME_HANDLER_MAP.keys()
    model_config = MODEL_NAME_HANDLER_MAP[name].get_config(config)
    return model_config


MODEL_NAME_HANDLER_MAP = {
    "llama2": Llama2BaseModelTokenizerHandler,
    "plllama": PrivateLlama,  # Private LoRA on LLaMA
}

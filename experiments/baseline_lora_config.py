LLAMA_QKV = [
    "q_proj",
    "k_proj",
    "v_proj",
]

LLAMA_ATTN = LLAMA_QKV + ["o_proj"]

LLAMA_MLP = [
    "up_proj",
    "gate_proj",
    "down_proj",
]

LLAMA_QKVUPGATE = LLAMA_QKV + [
    "up_proj",
    "gate_proj",
]

LLAMA_FULL = LLAMA_ATTN + LLAMA_MLP

LLAMA_FULL_CONFIG = [LLAMA_ATTN + LLAMA_MLP, None]

LORA_CONFIG_MAPS = {
    "llama2-7b": {
        "full": LLAMA_FULL_CONFIG,
        "loras": [
            LLAMA_QKV,
            [15, 31],
        ],
        "lorap": [  #
            LLAMA_QKV,
            list(range(0, 32)),
        ],
    },
    "llama-7b": {
        "full": LLAMA_FULL_CONFIG,
    },
    "llama2-13b": {
        "full": LLAMA_FULL_CONFIG,
        "lorap": [LLAMA_QKV, None],
        "loras": [LLAMA_QKV, [19, 39]],
    },
    "llama-30b": {
        "full": LLAMA_FULL_CONFIG,
        "qkv": [LLAMA_QKV, None],
        "loras": [LLAMA_QKV, [29, 59]],
        "lorap": [LLAMA_QKV, None],
    },
}


def get_lora_config(model, config_ind):
    return LORA_CONFIG_MAPS[model][config_ind]

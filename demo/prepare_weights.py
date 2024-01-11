import torch
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "llama_dir",
    required=True,
    default=None,
    help="root dir of llama weights, assuming weights are sharded into 3",
)
parser.add_argument(
    "pl_path",
    required=True,
    default=None,
    help="PrivateLoRA weights trained on GSM8K, download it from https://huggingface.co/wanglamao/PrivateLoRA_GSM8K",
)
parser.add_argument(
    "output_dir", required=True, default=None, help="dir for prepared weights"
)

args = parser.parse_args()

if __name__ == "__main__":
    # load llama2 7b state dict
    print("loading llama 7b weights...")

    sd7b = torch.load(
        os.path.join(args.llama_dir, "pytorch_model-00001-of-00003.bin"),
        map_location="cpu",
    )
    sd7b.update(
        torch.load(
            os.path.join(args.llama_dir, "pytorch_model-00002-of-00003.bin"),
            map_location="cpu",
        )
    )
    sd7b.update(
        torch.load(
            os.path.join(args.llama_dir, "pytorch_model-00003-of-00003.bin"),
            map_location="cpu",
        )
    )

    print("llama 7b weights loaded")
    print("loading privatelora gsm8k weights")
    gsm8ksd = torch.load(args.pl_path, map_location="cpu")
    print("privatelora gsm8k weights loaded")

    print("processing weights...")

    device_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    device_dict = {}
    cloud_dict = {}
    # split llama weights into cloud weights and device weights
    for k, v in sd7b.items():
        if k in device_keys:
            device_dict[k] = v
        elif "_proj" in k or "_layernorm" in k or "model.norm" in k:
            cloud_dict[k] = v
        else:
            print(k)

    # rename key for word embedding
    v = device_dict.pop("model.embed_tokens.weight")
    device_dict["embed_tokens.weight"] = v
    print(len(device_dict.keys()))
    print(len(cloud_dict.keys()))

    # split privatelora weights into cloud and device
    for i in range(32):
        aqkey = f"model.layers.{i}.self_attn.q_lora.lora_A.weight"
        akkey = f"model.layers.{i}.self_attn.k_lora.lora_A.weight"
        avkey = f"model.layers.{i}.self_attn.v_lora.lora_A.weight"
        new_akey = f"model.layers.{i}.self_attn.lora_AB.lora_A"
        bqkey = f"model.layers.{i}.self_attn.q_lora.lora_B.weight"
        bkkey = f"model.layers.{i}.self_attn.k_lora.lora_B.weight"
        bvkey = f"model.layers.{i}.self_attn.v_lora.lora_B.weight"
        new_bkey = f"model.layers.{i}.self_attn.lora_AB.lora_B"
        cloud_dict[new_akey] = torch.stack(
            [gsm8ksd[aqkey], gsm8ksd[akkey], gsm8ksd[avkey]]
        )
        cloud_dict[new_bkey] = torch.stack(
            [gsm8ksd[bqkey], gsm8ksd[bkkey], gsm8ksd[bvkey]]
        )

        mqkey = f"model.layers.{i}.self_attn.q_lora.lora_mobile.weight"
        mkkey = f"model.layers.{i}.self_attn.k_lora.lora_mobile.weight"
        mvkey = f"model.layers.{i}.self_attn.v_lora.lora_mobile.weight"
        new_mkey = f"lora_M_stack.layers.{i}.lora_M"
        device_dict[new_mkey] = torch.stack(
            [gsm8ksd[mqkey], gsm8ksd[mkkey], gsm8ksd[mvkey]]
        )

    # save processed weights
    print("Start saving weights")
    cloud_path = os.path.join(args.output_dir, "cloud/pytorch_model.bin")
    os.makedirs(os.path.dirname(cloud_path), exist_ok=True)
    torch.save(cloud_dict, cloud_path)
    print(f"cloud weight saved to {cloud_path}")

    device_path = os.path.join(args.output_dir, "device/pytorch_model.bin")
    os.makedirs(os.path.dirname(device_path), exist_ok=True)
    torch.save(device_dict, device_path)
    print(f"device weight saved to {device_path}")

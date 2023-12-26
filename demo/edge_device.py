from demo.model import PLLlamaConfig, LlamaForDevice
from pl_lib import CommProfiler
import torch
import logging
import argparse
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()

parser.add_argument(
    "weight_path",
    default=None,
    help="path to device model weight",
)
parser.add_argument(
    "llama_path",
    default=None,
    help="root dir of huggingface llama model, should contain weight files and config",
)
parser.add_argument(
    "--ip",
    default="127.0.0.1",
    help="socket ip of cloud",
)
parser.add_argument(
    "--port",
    default=12345,
    help="socket port of cloud",
)
parser.add_argument(
    "--device",
    default="cpu",
    help="device of model",
)
parser.add_argument(
    "--debug",
    default=False,
)
args = parser.parse_args()

log_format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO, format=log_format
)


if __name__ == "__main__":
    mock_small = True
    load_weights = False
    from pl_lib import init_tcp_b

    logging.info("start connecting...")
    s = init_tcp_b(args.ip, args.port)
    config = PLLlamaConfig.from_pretrained(args.llama_path)
    config.rcd = 128
    config.rdc = 128

    tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
    logging.info("Initializing Model")
    model = LlamaForDevice(config)
    model.set_tokenizer(tokenizer)
    print(model)
    logging.info("model ready")
    # Print param stats
    model.print_param_count()
    logging.info("loading weights")
    model.load_state_dict(torch.load(args.weight_path))
    logging.info("weights loaded")

    question = "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?"  # noqa
    input_ids = tokenizer(
        question,
        return_tensors="pt",
    ).input_ids

    # set profiling
    kwargs = {"s": s, "speed_profile": True, "comm_profiler": CommProfiler()}
    logging.info(f"input_ids {input_ids}")
    logging.info(f"question {question}")
    logging.info("query sent")
    # start generation
    outs = model.my_generate(
        input_ids=input_ids,
        max_new_tokens=250,
        **kwargs,
    )
    print(outs)
    print(tokenizer.batch_decode(outs))
    s.close()

# PrivateLoRA for GenAI Model Privatization

| [paper](https://arxiv.org/abs/2311.14030) | [weight](https://huggingface.co/wanglamao/PrivateLoRA_GSM8K) | [Blog](https://wanglamao.github.io/)
## Overview


<figure>
<img src="./doc/blog/privatelora.png" >
<figcaption style="text-align: center;"></figcaption>
</figure>

PrivateLoRA separates private compute from the model and deploy them on edge devices for data locality.
Network overhead is optimized to ensure throughput close to GPU on edge devices.

## Reproducing Experiment 

### Dependencies

```bash
pip install deepspeed==0.12.5 transformers==4.35.2 accelerate==0.25.0 peft==0.5.0 termplotlib
pip install -e ./lm-evaluation-harness
```

### Training


`scripts` folder contains launch scripts to train baseline methods and PrivateLoRA.

```bash
cd experiments
bash scripts/baseline.sh
bash scripts/pl.sh
```


### Evaluation

We write a custom callback based on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate during training.

To use our callback, firstly install our adapted version of `lm-eval`.
```
pip install -e ./lm-evaluation-harness
```

Then, import the callback and plug it in the trainer in your training scripts.
```python
from infras.eval import EvalHarnessCallBack
trainer.add_callback(
    EvalHarnessCallBack(
        trainer=trainer,              # huggingface trainer
        tokenizer=tokenizer,          # tokenizer
        tasks=["gsm8k_yaml","mmlu"],  # task name defined in `lm-eval`
        eval_steps=20,                # number of steps as evaluation interval
        eval_start=0,                 # evaluation start
        do_init_eval=False,           # evaluate before the first training step
        eval_batch_size=16, 
    )
)
```
Evaluation results will be logged in tensorboard during training. 

## Demo
To demonstrate that PrivateLoRA offers great user experience from model privatization, we make a runnable demo.


<figure>
<img src="./doc/blog/demo_gen.gif" >
<figcaption style="text-align: center;">Screenshots of Edge Device running on CPU.</figcaption>
</figure>

[Central Cloud](demo/central_cloud.py) hosts the decoder stack of LlaMA 2-7B and [Edge Device](demo/edge_device.py) hosts private parameters. Central Cloud runs on GPU and Edge Device runs on CPU.

User of Edge Device queries the model with a question picked from GSM8K. 
Since it's 0-shot test, the original LLaMA 2-7B can barely answer it correctly.
But parameters on "edge device" are trained on GSM8K, they will steer the model to correctly solve it!👍


Detailed instructions and explanations are documented [here](demo/readme.md).

## Citation

```bibtex
@article{privatelora,
  author       = {Yiming Wang and
                  Yu Lin and
                  Xiaodong Zeng and
                  Guannan Zhang},
  title        = {PrivateLoRA For Efficient Privacy Preserving {LLM}},
  journal      = {CoRR},
  volume       = {abs/2311.14030},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2311.14030},
  doi          = {10.48550/ARXIV.2311.14030},
  eprinttype    = {arXiv},
  eprint       = {2311.14030},
}
```


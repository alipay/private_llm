# Experiment Reproduction

This folder contains codes for training and evaluation reproduction.

## Usage

### Training

`scripts` folder contains training scripts of baseline methods and PrivateLoRA.

```bash
cd experiments
bash scripts/baseline.sh
```

### Evaluation

Evaluation is integrated into the training procedure and is triggered every `eval_steps`.

To use our callback, firstly install our adapted version of `lm-eval`.
```
pip install -e ./lm-evaluation-harness
```

Then, import the callback and plug it into your training scripts.
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
Evaluation results will be logged in tensorboard. 

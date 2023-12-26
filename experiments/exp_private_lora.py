# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
import os
import sys
from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
from typing import Optional
import logging
from infras.model_utils import get_base_model_and_tokenizer
from infras.arguments import MyExpConfig, ModelArguments, DataTrainingArguments
from infras.data import prepare_train_dataset
from dataclasses import dataclass, field, asdict


@dataclass
class PrivateLoraExpConfig(MyExpConfig):
    """arguments used in privatelora experiment"""

    is_baseline: Optional[bool] = field(
        default=False,
        metadata={"help": "if False then privatelora, otherwise lora or ft"},
    )
    baseline_use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "use lora"},
    )
    peft_lora_config_ind: Optional[str] = field(
        default=None,
        metadata={
            "help": "lora configuration for target modules and target layers, see baseline_lora_config.py"
        },
    )
    trainable_a: Optional[bool] = field(
        default=False,
        metadata={"help": "whether matrix a is trainable in privatelora"},
    )
    trainable_b: Optional[bool] = field(
        default=False,
        metadata={"help": "whether matrix b is trainable in privatelora"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, PrivateLoraExpConfig)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            model_args,
            data_args,
            training_args,
            private_lora_exp_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
            private_lora_exp_args,
        ) = parser.parse_args_into_dataclasses()

    # Log on each process the small summary:
    logging.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logging.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    # only privatelora m is trainable if it's private lora
    model, tokenizer = get_base_model_and_tokenizer(
        private_lora_exp_args.model_name,
    )

    # adapt with original lora
    if private_lora_exp_args.is_baseline and private_lora_exp_args.baseline_use_lora:
        from peft import get_peft_model, LoraConfig

        from baseline_lora_config import get_lora_config

        target_modules, layers_to_transform = get_lora_config(
            private_lora_exp_args.model_name,
            private_lora_exp_args.peft_lora_config_ind,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
        )

        model = get_peft_model(model, lora_config)

    # set a,b trainable
    if private_lora_exp_args.trainable_a:
        logging.info("Set lora a trainable")
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad = True
    if private_lora_exp_args.trainable_b:
        logging.info("Set lora b trainable")
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.requires_grad = True

    if training_args.do_train:
        from datasets import concatenate_datasets

        train_datasets = prepare_train_dataset(data_args, private_lora_exp_args)
        print(train_datasets)
        train_dataset = (
            concatenate_datasets(
                [train_dataset for train_dataset in train_datasets.values()]
            )
            if len(train_datasets.values()) > 1
            else list(train_datasets.values())[0]
        )
        if private_lora_exp_args.shuffle_data:
            train_dataset = train_dataset.shuffle()

    # Data collator
    from infras.data import CausalCollator

    collator_kwargs = {
        "target_max_len": data_args.target_max_len,
        "predict_with_generate": data_args.predict_with_generate,
        "train_on_source": data_args.train_on_source,
        "input_field": "input",
        "target_field": "output",
    }
    data_collator = CausalCollator(
        tokenizer, source_max_len=data_args.source_max_len, **collator_kwargs
    )

    # add callback

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # add some text records to tensorboard
    from transformers.integrations import TensorBoardCallback

    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, TensorBoardCallback):
            if training_args.local_rank == 0:
                if cb.tb_writer is None:
                    cb._init_summary_writer(args=training_args)
                cb.tb_writer.add_text(
                    tag="exp_param", text_string=str(asdict(private_lora_exp_args))
                )
                try:
                    cb.tb_writer.add_text(
                        tag="lora_config",
                        text_string=str(asdict(lora_config)),
                    )
                except Exception:
                    pass
                try:
                    private_lora_module = model.model.layers[0].self_attn.q_lora
                    cb.tb_writer.add_text(
                        tag="private_lora",
                        text_string=str(private_lora_module),
                    )
                except Exception:
                    pass

    if training_args.do_eval:
        logging.info("add eval callback")
        from infras.eval import EvalHarnessCallBack

        if training_args.eval_steps < 1.0:
            eval_steps = min(
                int(
                    len(train_dataset)
                    // (training_args.per_device_train_batch_size * 4)
                    * training_args.eval_steps
                ),
                1,
            )
        else:
            eval_steps = training_args.eval_steps
        trainer.add_callback(
            EvalHarnessCallBack(
                trainer=trainer,
                tokenizer=tokenizer,
                tasks=private_lora_exp_args.eval_tasks,
                eval_steps=eval_steps,
                eval_start=private_lora_exp_args.eval_start,
                do_init_eval=private_lora_exp_args.do_init_eval,
                eval_batch_size=training_args.per_device_eval_batch_size,
            )
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if training_args.local_rank == 0:
            logging.info(f"checkpoint = {checkpoint}")
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        logging.info("=" * 40)
        logging.info("Start Training ....")
        logging.info("=" * 40)
        trainer.train(resume_from_checkpoint=checkpoint)
        dist.barrier()
        save_on_zero_3(training_args, trainer, model)
    if training_args.deepspeed is not None:
        dist.barrier()


def save_on_zero_3(training_args, trainer, model):
    # check if zero3 mode enabled
    if training_args.hf_deepspeed_config.is_zero3():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        logging.info("start save state_dict_zero3")
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        state_dict = None
    if training_args.local_rank == 0:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            logging.info("save done")

        to_save = get_pl_dict(state_dict)
        if to_save is not None:
            import torch

            torch.save(to_save, f"{training_args.output_dir}/pl.bin")
            logging.info(f"save pl success to {training_args.output_dir}")


def get_pl_dict(state_dict):
    """return additional parameters (A,B,M) of private lora

    Args:
        state_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    to_save = {}
    v3_flag = False
    for k, v in state_dict.items():
        if "lora_mobile" in k:
            v3_flag = True
        if "lora_" in k:
            to_save[k] = v
    return to_save if v3_flag else None


if __name__ == "__main__":
    main()

from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass, field
import copy
import torch
from infras.arguments import DataTrainingArguments, MyExpConfig

IGNORE_INDEX = -100


@dataclass
class CausalCollator:
    """collator adapted from artido/qlora"""

    tokenizer: PreTrainedTokenizerBase
    source_max_len: int
    train_on_source: bool = field(default=False)
    input_field: str = field(default="input")
    target_field: str = field(default="output")
    target_max_len: int = field(default=10)
    predict_with_generate: bool = field(default=False)

    def __call__(self, features):
        # print(len(features))
        if not self.train_on_source:
            sources = [
                self.tokenizer.bos_token + feature[self.input_field]
                for feature in features
            ]
        else:
            sources = [
                self.tokenizer.bos_token
                + feature[self.input_field]
                + feature[self.target_field]
                + self.tokenizer.eos_token
                for feature in features
            ]
            max_length = self.source_max_len + self.target_max_len
            tokens = self.tokenizer(
                sources,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            labels = tokens["input_ids"].clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            tokens["labels"] = labels
            return tokens

        if self.predict_with_generate:
            tokens = self.tokenizer(
                sources,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.source_max_len,
            )
            tokens["idx"] = [feature["idx"] for feature in features]
            return tokens

        else:
            targets = []
            for feature in features:
                if isinstance(feature[self.target_field], list):
                    targets += [
                        ", ".join(feature[self.target_field]) + self.tokenizer.eos_token
                    ]
                else:
                    targets += [feature[self.target_field] + self.tokenizer.eos_token]

            # Tokenize
            tokenized_sources_with_prompt = self.tokenizer(
                sources,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_targets = self.tokenizer(
                targets,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            # Build the input and labels for causal LM
            input_ids = []
            labels = []
            for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt["input_ids"],
                tokenized_targets["input_ids"],
            ):
                if not self.predict_with_generate:
                    input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                    if not self.train_on_source:
                        labels.append(
                            torch.tensor(
                                [IGNORE_INDEX for _ in range(len(tokenized_source))]
                                + copy.deepcopy(tokenized_target)
                            )
                        )
                    else:
                        labels.append(
                            torch.tensor(
                                copy.deepcopy(tokenized_source + tokenized_target)
                            )
                        )
                else:
                    input_ids.append(torch.tensor(tokenized_source))
            # Apply padding
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = (
                pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
                if not self.predict_with_generate
                else None
            )
            data_dict = {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            }
            if labels is not None:
                data_dict["labels"] = labels
            # print(data_dict["input_ids"].shape)
            # for k in data_dict.keys():
            #     print(data_dict[k].shape)

            return data_dict


def prepare_train_dataset(
    data_args: DataTrainingArguments,
    private_lora_exp_args: MyExpConfig,
):
    """prepare train dataset

    Args:
        data_args (DataTrainingArguments): _description_
        private_lora_exp_args (MyExpConfig): _description_

    Returns:
        _type_: _description_
    """
    from infras.open_bench import get_datasets_to_use

    dataset_names = private_lora_exp_args.train_dataset_names
    datasets = get_datasets_to_use(dataset_names, "train")
    for k, dataset in datasets.items():
        print(dataset)
        if data_args.max_train_samples:
            datasets[k] = dataset.select(
                range(min(len(dataset), data_args.max_train_samples))
            )
    return datasets

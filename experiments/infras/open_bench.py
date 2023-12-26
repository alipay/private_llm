import os
import re
from datasets import load_from_disk, load_dataset
from dataclasses import dataclass
from typing import List

IGNORE_INDEX = -100


def parse_dataset_name_config(dataset_name: str):
    name, config = dataset_name.split("-")
    return name, config


def get_datasets_to_use(dataset_names: List[str], split):
    """_summary_

    Args:
        dataset_names (List[str]): 要用的数据集的名字，格式是 <数据集名称>-<子集名称>

        split (_type_): `'train','validation','test'`

    Returns:
        Dict[dataset]: key is dataset_name
    """
    datasets = {}
    for dataset_name in dataset_names:
        name, config = parse_dataset_name_config(dataset_name)
        datasets[dataset_name] = DATASET_NAME_HANDLER_MAP[name].get_dataset(
            config, split
        )
    return datasets


@dataclass
class BaseDatasetHandler:
    dataset_root_dir: str
    available_configs: List[str]
    available_splits: List[str]

    def format_dataset(self, config, dataset):
        return dataset

    def get_dataset_path(self, config=None):
        return os.path.join(self.dataset_root_dir, config)

    def load_dataset(self, config=None):
        if config not in self.available_configs:
            raise ValueError()
        path = self.get_dataset_path(self, config)
        return load_from_disk(path)

    @classmethod
    def get_dataset(cls, config: str, split):
        if split not in cls.available_splits:
            raise ValueError()
        if config not in cls.available_configs:
            raise ValueError()
        dataset = cls.load_dataset(cls, config)[split]
        dataset = cls.format_dataset(cls, config, dataset)
        return dataset

    @classmethod
    def show_all_configs(cls):
        return cls.available_configs


@dataclass
class SuperGLUEDatasetHandler(BaseDatasetHandler):
    dataset_root_dir = "/mntnlp/zhuangyou/datasets/super_glue"
    available_configs = ["boolq", "multirc", "cb", "rte", "wic", "wsc"]
    available_splits = ["train", "validation", "test"]

    def format_dataset(self, config, dataset):
        if config == "boolq":
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["passage"]} {x["question"][0].upper() + x["question"][1:]}{"" if x["question"].endswith("?") else "?"}\n',  # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=["question", "passage", "idx", "label"],
            )
        elif config == "multirc":
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"]["answer"],
                    "input": f'{x["paragraph"]}\nQuestion: {x["question"]}\nI found this answer "{x["answer"]}". Is that correct? Yes or No?\n',  # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=["paragraph", "question", "answer", "idx", "label"],
            )
        elif config == "copa":
            capitalization: str = "correct"
            effect_conj: str = " so "
            cause_conj: str = " because "
            outputs = None

            def get_conjucture(sample):
                if sample["question"] == "effect":
                    conjunction = effect_conj
                elif sample["question"] == "cause":
                    conjunction = cause_conj
                else:
                    raise NotImplementedError
                return conjunction

            def get_prompt(sample):
                premise = sample["premise"].rstrip()
                if premise.endswith(
                    "."
                ):  # TODO Add other scripts with different punctuation
                    premise = premise[:-1]
                conjunction = get_conjucture(sample)
                prompt = premise + conjunction
                if capitalization == "upper":
                    prompt = prompt.upper()
                elif capitalization == "lower":
                    prompt = prompt.lower()
                return prompt

            def encode(sample):
                prompt = get_prompt(sample)
                return prompt

            def capitalize(c):
                if capitalization == "correct":
                    words = c.split(" ")
                    if words[0] != "I":
                        words[0] = words[0].lower()
                    return " ".join(words)
                elif capitalization == "bug":
                    return c
                elif capitalization == "upper":
                    return c.upper()
                elif capitalization == "lower":
                    return c.lower()
                else:
                    raise NotImplementedError

            def verbalize(sample, candidate):
                prompt = get_prompt(sample)
                return prompt + capitalize(candidate)

            def encode_sfc(sample):
                conjunction = get_conjucture(sample)
                return conjunction.strip()

            def verbalize_sfc(sample, candidate):
                conjunction = get_conjucture(sample)
                sfc_prompt = conjunction.strip() + " " + capitalize(candidate)
                return sfc_prompt

            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": get_prompt(x),
                    "output": capitalize(x[f"choice{x['label']+1}"]),
                    "labels": x["label"],
                }
            )
        elif config == "record":
            outputs = None
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["passage"]}\n{x["query"]}\nQuestion: what is the "@placeholder"\nAnswer: ',
                    "output": x["answers"],
                    "labels": None,
                }
            )
        elif config == "rte":
            outputs = {0: "Yes", 1: "No"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["premise"]}\nDoes this mean that "{x["hypothesis"]}" is true? Yes or No?\n',
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=["premise", "hypothesis", "idx", "label"],
            )
        elif config == "cb":
            outputs = {0: "Yes", 1: "No", 2: "Maybe"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'Suppose {x["premise"]} Can we infer that "{x["hypothesis"]}"? Yes, No, or Maybe?',
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                }
            )
        elif config in ["wic"]:
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'Does the word "{x["word"]}" have the same meaning in these two sentences? Yes, No?\n{x["sentence1"]}\n{x["sentence2"]}\n',  # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=[
                    "word",
                    "sentence1",
                    "sentence2",
                    "start1",
                    "start2",
                    "end1",
                    "end2",
                    "idx",
                    "label",
                ],
            )
        elif config in ["wsc"]:
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["text"]}\nIn the previous sentence, does the pronoun "{x["span2_text"].lower()}" refer to {x["span1_text"]}? Yes or No?\n',  # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                }
            )
        else:
            raise NotImplementedError()

        return dataset


@dataclass
class GSM8KDatasetHandler(BaseDatasetHandler):
    dataset_root_dir = "/mntnlp/zhuangyou/datasets/gsm8k"
    available_configs = ["train", "test"]
    available_splits = ["train", "test"]

    def get_dataset_path(self, config=None):
        return self.dataset_root_dir

    def format_dataset(self, config, dataset):
        dataset = dataset.map(
            lambda x: {
                "input": "Question: {}\nAnswer:".format(x["question"]),
                "output": x["answer"],
            },
            remove_columns=["question", "answer"],
        )
        return dataset


@dataclass
class MMLUDatasetHandler(BaseDatasetHandler):
    available_configs = ["train", "fs", "zs"]
    available_splits = ["train", "validation", "test"]

    def format_dataset(self, config, dataset):
        if config != "train":
            return dataset

        def format_mmlu(sample):
            ins = sample["question"]
            if not ins.endswith("?"):
                ins += "____"
            abcd = sample["choices"]
            ins += f"\nA.{abcd[0]} B.{abcd[1]} C.{abcd[2]} D.{abcd[3]}\nAnswer:"
            op_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
            return {"input": ins, "output": op_dict[sample["answer"]]}

        return dataset.map(
            format_mmlu, remove_columns=["question", "choices", "answer"]
        )

    def load_dataset(self, config):
        if config == "zs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "validation": os.path.join(
                        "/mntnlp/zhuangyou/datasets/qlora_mmlu_eval/zero_shot_mmlu_val.json"
                    ),
                    "test": os.path.join(
                        "/mntnlp/zhuangyou/datasets/qlora_mmlu_eval/zero_shot_mmlu_test.json"
                    ),
                },
            )
        # MMLU Five-shot (Eval/Test only)
        elif config == "fs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "validation": os.path.join(
                        "/mntnlp/zhuangyou/datasets/qlora_mmlu_eval/five_shot_mmlu_val.json"
                    ),
                    "test": os.path.join(
                        "/mntnlp/zhuangyou/datasets/qlora_mmlu_eval/five_shot_mmlu_test.json"
                    ),
                },
            )
        elif config == "train":
            mmlu_dataset = {
                "train": load_from_disk(
                    "/mntnlp/zhuangyou/datasets/mmlu_auxiliary_train"
                )
            }
        return mmlu_dataset


@dataclass
class PIQADatasetHandler(BaseDatasetHandler):
    dataset_root_dir = "/mntnlp/zhuangyou/datasets/piqa"
    available_configs = ["zs", "train"]
    available_splits = ["train", "validation", "test"]
    option_dict = {
        "fs": {0: "A", 1: "B"},
        "zs": {0: "A", 1: "B"},
        "train": {0: "A", 1: "B"},
    }

    def format_dataset(self, config, dataset):
        def format_sample(sample):
            temp = {}
            temp["input"] = "Question: {}\nA. {}\nB. {}\nAnswer:".format(
                sample["goal"], sample["sol1"], sample["sol2"]
            )
            temp["output"] = self.option_dict[config][sample["label"]]
            return temp

        dataset = dataset.map(format_sample)
        return dataset

    def get_dataset_path(self, config=None):
        return self.dataset_root_dir


@dataclass
class HellaswagDatasetHandler(BaseDatasetHandler):
    dataset_root_dir = "/mntnlp/zhuangyou/datasets/hellaswag"
    available_configs = ["zs", "train"]
    available_splits = ["train", "validation", "test"]
    option_dict = {
        "zs": {0: "A", 1: "B", 2: "C", 3: "D"},
        "train": {0: "A", 1: "B", 2: "C", 3: "D"},
    }

    def format_dataset(self, config, dataset):
        def preprocess(text):
            text = text.strip()
            # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
            text = text.replace(" [title]", ". ")
            text = re.sub("\\[.*?\\]", "", text)
            text = text.replace("  ", " ")
            return text

        def format_sample(sample):
            ctx = sample["ctx_a"] + " " + sample["ctx_b"].capitalize()
            temp = {}
            temp["input"] = "{}: {} ___\n{}\nAnswer:".format(
                sample["activity_label"],
                ctx,
                "\n".join(
                    [
                        "{}. {}".format(self.option_dict[config][i], preprocess(t))
                        for i, t in enumerate(sample["endings"])
                    ]
                ),
            )
            temp["output"] = "{}. {}".format(
                self.option_dict[config][int(sample["label"])],
                sample["endings"][int(sample["label"])],
            )
            return temp

        dataset = dataset.map(format_sample)
        return dataset

    def get_dataset_path(self, config=None):
        return self.dataset_root_dir

    # def load_dataset(self, config):
    #     if config not in self.available_configs:
    #         raise ValueError()
    #     path = os.path.join(self.dataset_root_dir)
    #     return load_from_disk(path)


DATASET_NAME_HANDLER_MAP = {
    "super_glue": SuperGLUEDatasetHandler,
    "mmlu": MMLUDatasetHandler,
    "gsm8k": GSM8KDatasetHandler,
    "piqa": PIQADatasetHandler,
    "hellaswag": HellaswagDatasetHandler,
}

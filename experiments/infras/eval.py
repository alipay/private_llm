import os
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from typing import List
import random
import lm_eval
from lm_eval.models.huggingface import LoadedHFLM
from lm_eval.utils import (
    run_task_tests,
    get_git_commit_hash,
)
from lm_eval.evaluator import evaluate
from lm_eval.logger import eval_logger

IGNORE_INDEX = -100


class BaseEvalCallback(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        tokenizer,
        eval_steps=None,
        eval_start=None,
        do_init_eval=False,
    ) -> None:
        """base evaluation callback to control when to do the evaluation.

        Args:
            trainer (Trainer): _description_
            tokenizer (_type_): _description_
            eval_steps (_type_, optional): eval interval. Defaults to None.
            eval_start (_type_, optional): which step to start eval. Defaults to None.
            do_init_eval (bool, optional): eval before model training. Defaults to False.
        """
        if eval_steps is None:
            eval_steps = 1
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.eval_start = eval_start if eval_start is not None else 0
        self.do_init_eval = do_init_eval

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 0 and self.do_init_eval:
            self.evaluate(args, state, control, **kwargs)
        if (
            state.global_step % self.eval_steps == 0
            and state.global_step != 0
            and state.global_step >= self.eval_start
        ):
            self.evaluate(args, state, control, **kwargs)

    def evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass


class EvalHarnessCallBack(BaseEvalCallback):
    def __init__(
        self,
        trainer: Trainer,
        tokenizer,
        tasks: List[str],
        eval_steps=None,
        eval_start=None,
        do_init_eval=False,
        eval_batch_size=32,
    ) -> None:
        """This callback integrates Eleuther/lm-evaluation-harness into the training loop

        Args:
            trainer (Trainer): trainer
            tokenizer (_type_): tokenizer
            tasks (List[str]): evaluation task name, pls refer to yaml files of lm-evaluation-harness.
            eval_steps (_type_, optional): eval interval. Defaults to None.
            eval_start (_type_, optional): which step to start eval. Defaults to None.
            do_init_eval (bool, optional): eval before model training. Defaults to False.
            eval_batch_size (int, optional):  Defaults to 32.

        """
        super().__init__(
            trainer,
            tokenizer,
            eval_steps,
            eval_start,
            do_init_eval,
        )
        self.tasks = tasks
        self.eval_batch_size = eval_batch_size

    def evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.trainer.model.eval()

        lm = LoadedHFLM(
            model=self.trainer.model,
            tokenizer=self.tokenizer,
            batch_size=self.eval_batch_size,
            max_batch_size=128,
        )
        res = simple_evaluate(
            model=lm,
            tasks=self.tasks,
            # num_fewshot=0,
            use_cache=None,
            log_samples=self.log_samples,
            # limit=self.eval_batch_size*2,
        )
        if args.local_rank == 0:
            self.trainer.log(self.format_metrics_for_tb(res["results"]))
            print("trainer log done")
        self.trainer.model.train()

        print("evaluate done")

    def format_metrics_for_tb(self, results):
        res = {}
        for task, metrics in results.items():
            for metric_name, value in metrics.items():
                res[f"{task}-{metric_name}"] = value
        return res


def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=None,
    batch_size=None,
    max_batch_size=None,
    device=None,
    use_cache=None,
    limit=None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
):
    """This code snipet is taken from lm-evaluation-harness to adapt for loaded model in a training script.

    Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(
        1234
    )  # TODO: this may affect training runs that are run with evaluation mid-run.

    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    assert isinstance(model, lm_eval.api.model.LM)
    lm = model

    if use_cache is not None:
        print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank" + str(lm.rank) + ".db",
        )

    task_dict = lm_eval.tasks.get_task_dict(tasks)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            # if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue

        config = task_obj._config
        if num_fewshot is not None:
            if config["num_fewshot"] > 0:
                default_num_fewshot = config["num_fewshot"]
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )

            task_obj._config["num_fewshot"] = num_fewshot

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
    )
    if lm.rank == 0:
        # add info about the model and few shot config
        results["config"] = {
            "model": model
            if isinstance(model, str)
            else model.model.config._name_or_path,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values())
            if hasattr(lm, "batch_sizes")
            else [],
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
        }
        results["git_hash"] = get_git_commit_hash()
        return results
    else:
        return None

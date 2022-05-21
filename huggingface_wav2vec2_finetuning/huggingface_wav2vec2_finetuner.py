#!/usr/bin/env python3
import json
import logging
import os
import shutil

import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from typing import Union, Optional, Any

import wandb

from data_io.readwrite_files import read_json

from huggingface_wav2vec2_finetuning.base_model_for_finetuning import (
    BaseModelForFinetuning,
)
from huggingface_wav2vec2_finetuning.ctc_trainer import CTCTrainer
from huggingface_wav2vec2_finetuning.data_loading_commons import IterableDatasetBase
from huggingface_wav2vec2_finetuning.hf_finetune_utils import setup_logging
from misc_utils.buildable import Buildable
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import (
    deserialize_dataclass,
    UNDEFINED,
    _UNDEFINED,
    CLASS_REF_NO_INSTANTIATE,
    serialize_dataclass,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from misc_utils.hpc_computing_args import ClusterArgs

import datasets
import numpy as np
import torch
from packaging import version
from transformers import (
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    Wav2Vec2CTCTokenizer,
    HfArgumentParser,
)
from transformers.trainer_utils import is_main_process


torch.set_num_threads(4)

if is_apex_available():
    pass

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True

logger = logging.getLogger(__name__)

if "WANDB_API_KEY" not in os.environ:
    print(f"could not find WANDB_API_KEY!!! -> disabling wandb")
    os.environ["WANDB_DISABLED"] = "true"


@dataclass
class TrainArgs(Buildable):
    """
    Buildable cause some arguments could be shape-shifting Buildables!
    """

    run_name: str  # only for wandb?
    is_dryrun: bool = False
    overwrite_output_dir: bool = True
    max_steps: int = 1_000_000_000
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 1e-5
    # TODO what lr_scheduler_type is recommended?
    lr_scheduler_type: str = "linear"  # goes linearly to zero, from 0 to max_steps, from learning_rate to 0 # see: https://github.com/huggingface/transformers/blob/1a66a6c67734f39e8dddc3c5635b8b845f748c39/src/transformers/training_args.py#L173
    warmup_steps: int = 500
    evaluation_strategy: str = "steps"
    save_steps: int = 100000
    eval_steps: int = 10000
    logging_steps: int = 100
    save_total_limit: int = 3
    dataloader_num_workers: int = 2
    no_cuda: bool = False
    fp16: bool = True
    group_by_length: bool = True
    ignore_data_skip: bool = True


@dataclass
class HFWav2vec2Finetuner(CachedData):
    """
    runs the finetuning-experiment, training + evaluation
    """

    BASE_PATH: str = "some-where"
    finetune_model: Union[_UNDEFINED, BaseModelForFinetuning] = UNDEFINED
    train_dataset: Optional[IterableDatasetBase] = None
    eval_dataset: Optional[torch.utils.data.Dataset] = None

    train_args: Union[_UNDEFINED, TrainArgs] = UNDEFINED
    overwrite_old_cache: bool = False

    hf_train_args: TrainingArguments = field(init=False, repr=False)

    cache_base: PrefixSuffix = field(
        default_factory=lambda: BASE_PATHES["finetune_training"]
    )

    @property
    def name(self):
        return self.train_args.run_name

    @property
    def output_dir(self):
        return self.prefix_cache_dir("output_dir")

    @property
    def _is_ready(self) -> bool:
        if self._found_dataclass_json():
            if self.overwrite_old_cache:
                shutil.rmtree(str(self.cache_dir))
                print(
                    f"cache clash! {self.dataclass_json=} already exists! -> removing it!"
                )
            else:
                assert False, f"cache clash! {self.dataclass_json=} already exists!"

        return self._was_built

    def _build_cache(self):
        error = None
        try:
            # TODO: not really working!
            # with open(self.prefix_cache_dir("stdout.txt"), "w") as stdout_f, open(
            #     self.prefix_cache_dir("stderr.txt"), "w"
            # ) as stderr_f, redirect_stdout(stdout_f), redirect_stdout(stderr_f):
            self._train_and_evaluate()
        except Exception as e:
            error = e
            print(f"\nerror during training: {e}\n")
            traceback.print_exc()
        finally:
            self.train_dataset.__exit__()
            if error is not None:
                raise error

    def _train_and_evaluate(self):
        parser = HfArgumentParser((TrainingArguments))
        self.hf_train_args = parser.parse_dict(
            asdict(self.train_args) | {"output_dir": self.output_dir}
        )[0]
        self.hf_train_args.do_train = self.train_dataset is not None
        self.hf_train_args.do_eval = self.eval_dataset is not None

        print(f"local-rank: {self.hf_train_args.local_rank}")
        # last_checkpoint = detecting_last_checkpoint(self.hf_train_args, logger)
        last_checkpoint = None  # TODO: have not yet tried to continue form checkpoint
        setup_logging(self.hf_train_args, logger, logging)

        wer_metric = datasets.load_metric("wer")

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[
                pred.label_ids == -100
            ] = self.finetune_model.processor.tokenizer.pad_token_id

            pred_str = self.finetune_model.processor.batch_decode(pred_ids)
            # we do not want to group tokens when computing the metrics
            label_str = self.finetune_model.processor.batch_decode(
                pred.label_ids, group_tokens=False
            )

            wer = wer_metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}

        trainer = CTCTrainer(
            model=self.finetune_model.model,
            data_collator=self.finetune_model.data_collator,
            args=self.hf_train_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.finetune_model.processor.feature_extractor,
            dryrun=self.train_args.is_dryrun,
        )
        # if is_dryrun:
        #     for _ in tqdm(
        #         itertools.islice(trainer.get_train_dataloader(), 0, 10_000),
        #         desc="dry-run dataloading",
        #     ):
        #         pass
        # assert False

        if self.hf_train_args.do_eval:
            logger.info("*** Initial Evaluate ***")
            metrics = trainer.evaluate()
            max_val_samples = len(self.eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(self.eval_dataset))

            trainer.log_metrics("initial-eval", metrics)
            trainer.save_metrics("initial-eval", metrics)

        if self.hf_train_args.do_train:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            # TODO: outcommented cause I don't want to continue from checkpoint
            # TODO: do I never ever want to continue from checkpoint?
            # elif os.path.isdir(model_args.model_name_or_path):
            #     checkpoint = model_args.model_name_or_path
            else:
                checkpoint = None

            if is_main_process(self.hf_train_args.local_rank):
                self.finetune_model.processor.save_pretrained(
                    self.hf_train_args.output_dir
                )

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()

            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if self.hf_train_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_val_samples = len(self.eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(self.eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        wandb.finish()


def main():
    finetuner: HFWav2vec2Finetuner = deserialize_dataclass(
        json.dumps(read_json(sys.argv[-1]))
    )
    finetuner.build()


if __name__ == "__main__":
    main()

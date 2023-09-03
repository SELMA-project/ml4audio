import functools
import os
from dataclasses import dataclass

import sys
from typing import Optional, Any

import numpy as np
import transformers
from beartype import beartype
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    set_seed,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from misc_utils.beartypes import NeNpFloatDim1, TorchTensor1D, NeStr

SILENCE_SYMBOL = "|"


def detecting_last_checkpoint(training_args, logger):
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
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logger.info(f"last_checkpoint: {last_checkpoint}")
    return last_checkpoint


def setup_logging(training_args, logger, logging):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)


TARGET_SAMPLE_RATE = 16_000


def get_len(datum):
    len_after_resampling = int((datum["end"] - datum["start"]) * TARGET_SAMPLE_RATE)
    return len_after_resampling


@dataclass
class HfASRSample:
    input_values: NeNpFloatDim1
    sampling_rate: int
    labels: list[int]


@beartype
def apply_asr_processor(
    audio: NeNpFloatDim1,
    text: str,  # NeStr here?
    feature_extractor: Wav2Vec2FeatureExtractor,
    tokenizer: Wav2Vec2CTCTokenizer,
) -> HfASRSample:
    """
    applies Wav2Vec2Processor
    :param audio: -> feature_extraction
    :param text: -> tokenization
    """
    input_values = feature_extractor(
        raw_speech=audio, sampling_rate=TARGET_SAMPLE_RATE
    ).input_values
    assert len(input_values) == 1
    input_values = input_values[0].squeeze()
    is_just_noise = len(text) == 0
    if is_just_noise:
        text = SILENCE_SYMBOL  # TODO: how to handle noise/silence, with space or | ?

    labels = tokenizer(text=text).input_ids
    return HfASRSample(
        input_values=input_values,
        sampling_rate=TARGET_SAMPLE_RATE,
        labels=labels,
    )


class NoModelSaveEarlyStoppingCallback(EarlyStoppingCallback):
    """
    original EarlyStoppingCallback monitors the state.best_metric which is only set when model is saved
    I want early stopping without having to save models all the time
    """

    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: Optional[float] = 0.0,
        min_steps: int = 1000,
    ):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.min_steps = min_steps
        self.best_value: Optional[float] = None

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if (
            state.global_step < self.min_steps
            or self.best_value is None
            or (
                operator(metric_value, self.best_value)
                and abs(metric_value - self.best_value) > self.early_stopping_threshold
            )
        ):
            self.best_value = metric_value
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            print(
                f"{self.__class__.__name__}: {self.early_stopping_patience_counter=} of {self.early_stopping_patience}"
            )


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    """
    TODO: I have not yet tested this!
        ReduceLROnPlateauWithWarmup nice idea, but is it really necessary/useful? see: https://github.com/huggingface/transformers/issues/16503
    """

    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = ...) -> None:
        if metrics is None:
            # trainer calls step-method without arguments
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group["lr"])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group["lr"] = new_lr
        else:
            super().step(metrics, epoch)


@beartype
def create_asr_vocabulary_for_training(
    vocab_set: list[str],
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
) -> dict[str, int]:
    """
    based on create_vocabulary_from_data
    """
    assert len(set(vocab_set)) == len(vocab_set)
    vocab_dict = {v: k for k, v in enumerate(vocab_set)}

    # replace white space with delimiter token
    if word_delimiter_token is not None and " " in vocab_dict:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None and unk_token not in vocab_dict:
        assert unk_token not in [x.upper() for x in vocab_dict] + [
            x.lower() for x in vocab_dict
        ]
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None and pad_token not in vocab_dict:
        assert pad_token not in [x.upper() for x in vocab_dict] + [
            x.lower() for x in vocab_dict
        ]
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict

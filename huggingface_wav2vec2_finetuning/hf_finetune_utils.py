import os
from dataclasses import dataclass

import sys
import transformers
from beartype import beartype
from transformers import (
    set_seed,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from misc_utils.beartypes import NumpyFloat1DArray, TorchTensor1D, NeStr

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
    input_values: NumpyFloat1DArray
    sampling_rate: int
    labels: list[int]


@beartype
def apply_asr_processor(
    audio: NumpyFloat1DArray,
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

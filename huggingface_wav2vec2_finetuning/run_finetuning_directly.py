import os
from dataclasses import dataclass
from typing import Iterator

from beartype import beartype

from huggingface_wav2vec2_finetuning.base_model_for_finetuning import (
    ModelArgs,
    ModelIdentity, DataArgs,
)
from huggingface_wav2vec2_finetuning.huggingface_wav2vec2_finetuner import (
    TrainArgs,
    HFWav2vec2Finetuner,
)
from huggingface_wav2vec2_finetuning.stream_ftdataset import IterableSlicingDataset
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.audio_data.mls_corpora import MLSIterableDataset, MLSTarGzTranscripts
from ml4audio.audio_data.targz_asr_dataset import TarGzArrayText
from ml4audio.audio_utils.audio_data_models import AudioTextData, ArrayText
from ml4audio.text_processing.asr_text_normalization import Casing


@beartype
def create_finetuner(
    run_name_for_wandb: str,
    model_to_finetune: ModelIdentity,
    train_corpus: AudioTextData,
    eval_corpus: AudioTextData,
    do_normalize_audio: bool,
):
    augmentations = [
        # TranscodePerturbationDC(0.5),
        # SoxPerturbations(proba=0.75),
    ]
    # fmt: off
    new_vocab = ["<pad>", "<s>", "</s>","<unk>", "|", "'", "-", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "ä", "ö", "ü","ß"]
    # fmt: on

    model_args = ModelArgs(
        model_to_finetune=model_to_finetune,
        text_normalizer="de",
        # casing=Casing.upper
        casing=Casing.lower,
        new_vocab=new_vocab,
        do_normalize_audio=do_normalize_audio,
        # attention_dropout=0.1,
        # activation_dropout=0.1,
        # feat_proj_dropout=0.1,
        # hidden_dropout=0.1,
        # final_dropout=0.0,
        # mask_time_prob=0.05,
    )
    data_args=DataArgs(
        eval_metrics=["wer","cer"]
    )
    finetune_task = HFWav2vec2Finetuner(
        model_args=model_args,
        data_args=data_args,
        train_dataset=IterableSlicingDataset(
            array_texts=train_corpus,
            perturbations=augmentations,
            limit=100_000,
        ),
        eval_dataset=IterableSlicingDataset(array_texts=eval_corpus, limit=100),
        train_args=TrainArgs(
            run_name=run_name_for_wandb,
            overwrite_output_dir=True,
            max_steps=20_000,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            learning_rate=1.0e-04,
            lr_scheduler_type="constant_with_warmup",
            # lr_scheduler_type="constant",
            warmup_steps=2000,
            evaluation_strategy="steps",
            save_steps=1000,
            eval_steps=100,
            # logging_steps=5,
            logging_steps=10,
            save_total_limit=3,
            dataloader_num_workers=0,
            no_cuda=False,
            fp16=True,
            group_by_length=False,
            ignore_data_skip=True,
            min_steps=10_000,
            gradient_accumulation_steps=4,
            deepspeed="huggingface_wav2vec2_finetuning/ds_config_zero3.json",
            # deepspeed="../ml4audio/huggingface_wav2vec2_finetuning/ds_config_zero3.json"

        ),
        overwrite_old_cache=True,
    )
    return finetune_task


if __name__ == "__main__":

    base_path = os.environ["BASE_PATH"]
    data_root = os.environ["DATA_ROOT"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["data_root"] = data_root
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["finetune_training"] = PrefixSuffix("cache_root", f"FINETUNE_TRAINING")
    BASE_PATHES["raw_data"] = PrefixSuffix(
        "cache_root", f"RAW_DATA"
    )  # TODO: I'm confused! when str ? when PrefixSuffix?
    BASE_PATHES["finetune_results"] = PrefixSuffix("cache_root", f"FINETUNE_RESULTS")
    BASE_PATHES["transformers_cache_dir"] = PrefixSuffix(
        "base_path", "huggingface_cache/transformers"
    )

    experiments = (
        finetune_task
        for model_to_finetune in [
            ModelIdentity(
                # "facebook/wav2vec2-base-960h",
                # "facebook/wav2vec2-base",
                "facebook/wav2vec2-xls-r-1b",
                # "jonatasgrosman/wav2vec2-large-xlsr-53-german"
            ),  # facebooks base model wants upper-cased vocab
        ]
        for eval_corpus in [
            TarGzArrayText(
                corpus=MLSIterableDataset(
                    targztranscripts=MLSTarGzTranscripts(
                        targz_file=str(PrefixSuffix("data_root", "mls_english.tar.gz"))
                    ),
                    split="test",
                ),
                sample_rate=16000,
            )
        ]
        for train_corpus in [
            TarGzArrayText(
                corpus=MLSIterableDataset(
                    targztranscripts=MLSTarGzTranscripts(
                        targz_file=str(PrefixSuffix("data_root", "mls_english.tar.gz"))
                    ),
                    split="train",
                ),
                sample_rate=16000,
            )
        ]
        for norm_audio in [True]
        for run_name in [f"debug_run_directly"]
        for finetune_task in [
            create_finetuner(
                run_name_for_wandb=run_name,
                model_to_finetune=model_to_finetune,
                train_corpus=train_corpus,
                eval_corpus=eval_corpus,
                do_normalize_audio=norm_audio,
            )
        ]
    )
    for exp in experiments:
        exp.build()
    # first_finetuned = FinetunedModel(finetune_master=o)

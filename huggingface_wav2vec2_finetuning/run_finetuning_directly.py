import sys

from beartype import beartype
#
# from data_loading.stream_ftdataset import IterableASRCorporaDataset
# from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
# from speech_data.asr_corpora import (
#     Auteda,
# )
# from speech_data.mls_corpora import MLSIterableDataset, MLSTarGzTranscripts
# from speech_data.targz_asr_dataset import TarGzArrayText
#
# from hf_wav2vec2_finetuning.finetune_huggingface_wav2vec2 import (
#     HFWav2vec2Finetuner,
#     TrainArgs,
# )
# from hf_wav2vec2_finetuning.finetune_model import FinetuneModel, ModelToFinetune
#
from huggingface_wav2vec2_finetuning.base_model_for_finetuning import \
    BaseModelForFinetuning, ModelIdentity


@beartype
def create_finetuner(
    run_name_for_wandb: str,
    model_to_finetune: ModelIdentity,
    train_corpus: Auteda,
    eval_corpus: Auteda,
    do_normalize_audio: bool,
):
    augmentations = [
        # TranscodePerturbationDC(0.5),
        # SoxPerturbations(proba=0.75),
    ]
    fm = BaseModelForFinetuning(
        model_to_finetune=model_to_finetune,
        text_normalizer="de",
        new_vocab=None,
        do_normalize_audio=do_normalize_audio,
    )
    finetune_task = HFWav2vec2Finetuner(
        finetune_model=fm,
        train_dataset=IterableASRCorporaDataset(
            corpus=train_corpus,
            finetune_model=fm,
            perturbations=augmentations,
            limit=100_000,
        ),
        eval_dataset=IterableASRCorporaDataset(
            corpus=eval_corpus, finetune_model=fm, limit=100
        ),
        train_args=TrainArgs(
            run_name=run_name_for_wandb,
            overwrite_output_dir=True,
            max_steps=10_000,
            num_train_epochs=1,
            # per_device_train_batch_size=6,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=1,
            learning_rate=1.0e-05,
            warmup_steps=500,
            evaluation_strategy="steps",
            save_steps=20_000,
            eval_steps=20_000,
            # save_steps=10,
            # eval_steps=10,
            # logging_steps=5,
            logging_steps=100,
            save_total_limit=3,
            dataloader_num_workers=4,
            no_cuda=False,
            fp16=True,
            group_by_length=False,
            ignore_data_skip=True,
        ),
        overwrite_old_cache=True,
    )
    return finetune_task


if __name__ == "__main__":

    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["data_root"] = "/p/project/selma-test/data"
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["finetune_training"] = PrefixSuffix("cache_root", f"FINETUNE_TRAINING")
    BASE_PATHES["raw_data"] = PrefixSuffix(
        "cache_root", f"RAW_DATA"
    )  # TODO: I'm confused! when str ? when PrefixSuffix?
    BASE_PATHES["finetune_results"] = PrefixSuffix("cache_root", f"FINETUNE_RESULTS")

    experiments = (
        finetune_task
        for model_to_finetune in [
            ModelToFinetune(
                "jonatasgrosman/wav2vec2-large-xlsr-53-german"
            ),
        ]
        for eval_corpus in [
            TarGzArrayText(
                corpus=MLSIterableDataset(
                    targztranscripts=MLSTarGzTranscripts(
                        targz_file=str(PrefixSuffix("data_root", "mls_german.tar.gz"))
                    ),
                    split="test",
                ),
            )
        ]
        for train_corpus in [
            TarGzArrayText(
                corpus=MLSIterableDataset(
                    targztranscripts=MLSTarGzTranscripts(
                        targz_file=str(PrefixSuffix("data_root", "mls_german.tar.gz"))
                    ),
                    split="train",
                ),
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

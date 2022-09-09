import os
from dataclasses import dataclass
from typing import Iterator

from beartype import beartype

from huggingface_wav2vec2_finetuning.base_model_for_finetuning import (
    ModelArgs,
    ModelIdentity,
    DataArgs,
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
    data_args = DataArgs(eval_metrics=["wer", "cer"])
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
            max_steps=200,
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
            # deepspeed="huggingface_wav2vec2_finetuning/ds_config_zero3.json", # TODO: was not working?
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
                "facebook/wav2vec2-base",
                # "facebook/wav2vec2-xls-r-1b",
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

"""
# output should look similar to this: 


extract_transcript_files from .../Multilingual_LibriSpeech/mls_english.tar.gz: 2it [00:16,  8.38s/it]found all transcripts-files!                                                                                                                                                                    
extract_transcript_files from .../Multilingual_LibriSpeech/mls_english.tar.gz: 2it [00:55, 27.95s/it]                                                                                                                                                                                                
build_self of MLSTarGzTranscripts took:167.98549032211304 seconds                                                                                             
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).                                                                                                                       
local-rank: -1                                                                 
09/09/2022 11:45:46 - WARNING - huggingface_wav2vec2_finetuning.huggingface_wav2vec2_finetuner - Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: True                                                                                                                               
09/09/2022 11:45:46 - INFO - huggingface_wav2vec2_finetuning.huggingface_wav2vec2_finetuner - Training/evaluation parameters TrainingArguments(                                                                                                                                                                              
_n_gpu=1,                                                                      
adafactor=False,                                                               
adam_beta1=0.9,                                                                
adam_beta2=0.999,                                                              
adam_epsilon=1e-08,                                                            
auto_find_batch_size=False,                                                    
bf16=False,                                                                    
bf16_full_eval=False,                                                          
data_seed=None,                                                                
dataloader_drop_last=False,                                                    
dataloader_num_workers=0,                                                      
dataloader_pin_memory=True,                                                    
ddp_bucket_cap_mb=None,                                                        
ddp_find_unused_parameters=None, 

... 

***** initial-eval metrics *****
  eval_cer                =     2.2593
  eval_loss               =    12.2895
  eval_runtime            = 0:00:07.93
  eval_samples            =        100
  eval_samples_per_second =     12.595
  eval_steps_per_second   =     12.595
  eval_wer                =        1.0
***** Running training *****
  Num examples = 100000
  Num Epochs = 1
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 4
  Total optimization steps = 200

{'loss': 11.0863, 'learning_rate': 5.000000000000001e-07, 'epoch': 0.0}                                                                                                                                            
{'loss': 10.2139, 'learning_rate': 1.0000000000000002e-06, 'epoch': 0.0}                                                                                                                                           

...

{'loss': 3.1954, 'learning_rate': 7.95e-06, 'epoch': 0.03}                                                                                                                                                         
{'loss': 3.1072, 'learning_rate': 8.45e-06, 'epoch': 0.03}                                                                                                                                                         
{'loss': 3.0851, 'learning_rate': 8.95e-06, 'epoch': 0.03}                                                                                                                                                         
worker_idx=0,self.local_rank=0: consumer_stats={'data_consuming_speed': 86.16219875366991, 'data_processing_speed': 77.0351257622233, 'overall_loading_speed': 40.671719567879535}
{'loss': 3.0799, 'learning_rate': 9.450000000000001e-06, 'epoch': 0.03}                                                                                                                                            
{'loss': 3.046, 'learning_rate': 9.950000000000001e-06, 'epoch': 0.03}                                                                                                                                             
100%|█████████████████| 200/200 [05:52<00:00,  1.85s/it]***** Running Evaluation *****
  Num examples = 100
  Batch size = 1

read 0 samples from mls_english-test, read-speed: 0.6247192765101427
The following columns in the evaluation set don't have a corresponding argument in `Wav2Vec2ForCTC.forward` and have been ignored: sampling_rate. If sampling_rate are not expected by `Wav2Vec2ForCTC.forward`,  you can safely ignore this message.
                                                                                                                                                                                                                  worker_idx=0,self.local_rank=0: consumer_stats={'data_consuming_speed': 6.703998102894713, 'data_processing_speed': 6.657164294536598, 'overall_loading_speed': 3.3402495586620393}| 11/100 [00:00<00:03, 23.91it/s]
{'eval_loss': 3.1090574264526367, 'eval_wer': 1.0, 'eval_cer': 1.0, 'eval_runtime': 5.4913, 'eval_samples_per_second': 18.211, 'eval_steps_per_second': 18.211, 'epoch': 0.03}                                     
                                                                                                                                                                                                                   
Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 357.8257, 'train_samples_per_second': 8.943, 'train_steps_per_second': 0.559, 'train_loss': 6.0282745456695555, 'epoch': 0.03}                                                                   

...

***** eval metrics *****
  epoch                   =       0.03
  eval_cer                =        1.0
  eval_loss               =     3.1091
  eval_runtime            = 0:00:05.52
  eval_samples            =        100
  eval_samples_per_second =     18.085
  eval_steps_per_second   =     18.085
  eval_wer                =        1.0
build_self of HFWav2vec2Finetuner took:406.9063937664032 seconds

"""
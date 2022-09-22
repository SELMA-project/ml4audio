# based on: https://github.com/huggingface/transformers/blob/220da3b8a1cde5870696369a02227f9211d626be/examples/research_projects/robust-speech-event/run_speech_recognition_ctc_bnb.py
import json
import logging
import os
import shutil
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from typing import Union, Optional

import numpy as np
import torch
from datasets import load_metric
from packaging import version
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    TrainingArguments,
    Wav2Vec2Processor,
    is_apex_available,
    Wav2Vec2CTCTokenizer,
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModelForCTC,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import is_main_process

from data_io.readwrite_files import read_json
from huggingface_wav2vec2_finetuning.base_model_for_finetuning import (
    ModelArgs,
    DataArgs,
)
from huggingface_wav2vec2_finetuning.ctc_data_collator import DataCollatorCTCWithPadding
from huggingface_wav2vec2_finetuning.ctc_trainer import CTCTrainer
from huggingface_wav2vec2_finetuning.data_loading_commons import IterableDatasetBase
from huggingface_wav2vec2_finetuning.hf_finetune_utils import (
    setup_logging,
    NoModelSaveEarlyStoppingCallback,
    create_asr_vocabulary_for_training,
)
from misc_utils.buildable import Buildable
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import (
    deserialize_dataclass,
    UNDEFINED,
    _UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer

torch.set_num_threads(4)  # TODO(tilo): why?

# if is_apex_available():
#     pass

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True

logger = logging.getLogger(__name__)

if "WANDB_API_KEY" not in os.environ:
    print(f"could not find WANDB_API_KEY!!! -> disabling wandb")
    os.environ["WANDB_DISABLED"] = "true"
    wandb = None
else:
    import wandb


@dataclass
class TrainArgs(Buildable):
    """
    based on DataTrainingArguments
    is used to created hf's TrainingArguments
    field names must match with hf's TrainingArguments!
    Buildable cause some arguments could be shape-shifting Buildables!
    """

    run_name: str  # only for wandb?
    # is_dryrun: bool = False # TODO: remove?
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
    do_train: bool = True
    do_eval: bool = True
    metric_for_best_model: str = "wer"  # eval_wer
    load_best_model_at_end: bool = True
    greater_is_better: bool = False
    early_stopping_patience: Optional[int] = 2
    early_stopping_threshold: float = 0.001
    min_steps: int = 10_000
    gradient_accumulation_steps: int = 1
    deepspeed: Optional[str] = None  # "ds_config_zero3.json"

    def __post_init__(self):
        assert (
            self.save_steps % self.eval_steps == 0
        ), f"hf-transformers says: load_best_model_at_end requires the saving steps to be a round multiple of the evaluation steps"


def _prepare_models(
    data_args: DataArgs,
    model_args: ModelArgs,
    training_args: TrainingArguments,
):

    # save special tokens for tokenizer
    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=data_args.use_auth_token,
    )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_kwargs = {}
    if model_args.new_vocab is not None:
        tokenizer_name_or_path = training_args.output_dir

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

        # with training_args.main_process_first():
        #     if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
        #         os.remove(vocab_file)

        with training_args.main_process_first(desc="new vocab"):
            # if not os.path.isfile(vocab_file):
            os.makedirs(tokenizer_name_or_path, exist_ok=True)
            vocab_dict = create_asr_vocabulary_for_training(
                model_args.new_vocab,
                word_delimiter_token=data_args.word_delimiter_token,
                unk_token=data_args.unk_token,
                pad_token=data_args.pad_token,
            )
            # save vocab dict to be loaded into tokenizer
            with open(vocab_file, "w") as file:
                json.dump(vocab_dict, file)

        # if tokenizer has just been created
        # it is defined by `tokenizer_class` if present in config else by `model_type`
        tokenizer_kwargs = {
            "config": config if config.tokenizer_class is not None else None,
            "tokenizer_type": config.model_type
            if config.tokenizer_class is None
            else None,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "word_delimiter_token": word_delimiter_token,
        }
        original_model = AutoModelForCTC.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=str(BASE_PATHES["transformers_cache_dir"]),
            # vocab_size=len(self.processor.tokenizer),
        )
        state_dict = original_model.state_dict()
        state_dict.pop("lm_head.weight")
        state_dict.pop("lm_head.bias")
    else:
        tokenizer_name_or_path = model_args.model_name_or_path
        state_dict = None

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_auth_token=data_args.use_auth_token,
        **tokenizer_kwargs,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=data_args.use_auth_token,
    )

    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        state_dict=state_dict,
        ignore_mismatched_sizes=state_dict is not None,
        cache_dir=model_args.cache_dir,
        config=config,
        use_auth_token=data_args.use_auth_token,
    )

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # Now save everything to be able to create a single processor later
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    try:
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)
        # Wav2Vec2Processor instead of AutoProcessor, otherwise it tries to load Wav2Vec2ProcessorWithLM here!
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    tokenizer: Wav2Vec2CTCTokenizer
    vocab = list(tokenizer.get_vocab().keys())

    transcript_normalizer = TranscriptNormalizer(
        casing=model_args.casing,
        text_normalizer=model_args.text_normalizer,
        vocab=vocab,
    )

    return model, processor, transcript_normalizer


# for deepspeed -> TODO: not yet working!
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"


@dataclass
class HFWav2vec2Finetuner(CachedData):
    """
    runs the finetuning-experiment, training + evaluation
    """

    BASE_PATH: str = "some-where"
    model_args: Union[_UNDEFINED, ModelArgs] = UNDEFINED
    data_args: Union[_UNDEFINED, DataArgs] = field(default_factory=lambda: DataArgs())
    train_dataset: Optional[IterableDatasetBase] = None
    eval_dataset: Optional[torch.utils.data.Dataset] = None

    train_args: Union[_UNDEFINED, TrainArgs] = UNDEFINED
    overwrite_old_cache: bool = False

    hf_train_args: TrainingArguments = field(init=False, repr=False, default=UNDEFINED)

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
        if self._found_cached_data():
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
        print(f"local-rank: {self.hf_train_args.local_rank}")
        # last_checkpoint = detecting_last_checkpoint(self.hf_train_args, logger)
        last_checkpoint = None  # TODO: have not yet tried to continue form checkpoint
        setup_logging(self.hf_train_args, logger, logging)

        model, processor, transcript_normalizer = _prepare_models(
            self.data_args, self.model_args, self.hf_train_args
        )
        tokenizer = processor.tokenizer
        self.train_dataset.set_things(
            processor.feature_extractor, tokenizer, transcript_normalizer
        )
        self.eval_dataset.set_things(
            processor.feature_extractor, tokenizer, transcript_normalizer
        )

        eval_metrics = {
            metric: load_metric(metric) for metric in self.data_args.eval_metrics
        }

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

            pred_str = tokenizer.batch_decode(pred_ids)
            # we do not want to group tokens when computing the metrics
            label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
            assert len(label_str) == len(
                pred_str
            ), f"{len(label_str)=}!={len(pred_str)=}"

            metrics = {
                k: v.compute(predictions=pred_str, references=label_str)
                for k, v in eval_metrics.items()
            }

            return metrics

        training_args = self.hf_train_args
        # Instantiate custom data collator
        data_collator = DataCollatorCTCWithPadding(processor=processor)

        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        if training_args.deepspeed is None:
            import bitsandbytes as bnb

            optimizer = bnb.optim.Adam8bit(
                params=optimizer_grouped_parameters,
                lr=training_args.learning_rate,
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_epsilon,
            )

            # TODO: ReduceLROnPlateauWithWarmup nice idea, but is it really necessary/useful? see: https://github.com/huggingface/transformers/issues/16503
            # scheduler = ReduceLROnPlateauWithWarmup(
            #     optimizer=optimizer,
            #     mode="min",
            #     factor=0.1,
            #     patience=2,
            # )
            optimizers = (optimizer, None)
        else:
            optimizers = (None, None)
            raise NotImplementedError("TODO: if oyu want it, fix it!")

        trainer = CTCTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=processor.feature_extractor,
            optimizers=optimizers,
        )
        if (
            self.train_args.early_stopping_patience is not None
            and self.train_args.early_stopping_patience > 0
        ):
            trainer.add_callback(
                NoModelSaveEarlyStoppingCallback(
                    self.train_args.early_stopping_patience,
                    self.train_args.early_stopping_threshold,
                    self.train_args.min_steps,
                )
            )

        if self.hf_train_args.do_eval:
            logger.info("*** Initial Evaluate ***")
            metrics = trainer.evaluate()
            max_val_samples = len(self.eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(self.eval_dataset))

            trainer.log_metrics("initial-eval", metrics)
            trainer.save_metrics("initial-eval", metrics)

        if training_args.do_train:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            # TODO: outcommented cause I don't want to continue from checkpoint
            # TODO: do I never ever want to continue from checkpoint?
            # elif os.path.isdir(model_args.model_name_or_path):
            #     checkpoint = model_args.model_name_or_path
            else:
                checkpoint = None

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()

            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_val_samples = len(self.eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(self.eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        if wandb is not None:
            wandb.finish()


def main():
    finetuner: HFWav2vec2Finetuner = deserialize_dataclass(
        json.dumps(read_json(sys.argv[-1]))
    )
    finetuner.build()


if __name__ == "__main__":
    main()

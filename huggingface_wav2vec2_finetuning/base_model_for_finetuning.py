import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Union, Optional

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
)
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import (
    Wav2Vec2FeatureExtractor,
)

from huggingface_wav2vec2_finetuning.ctc_data_collator import DataCollatorCTCWithPadding
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import (
    _UNDEFINED,
    UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.text_processing.asr_text_cleaning import Casing
from ml4audio.text_processing.character_mappings.text_cleaning import (
    TextCleaner,
)


@dataclass
class ModelIdentity:
    """
    formerly known as ModelToFinetune: name / handle / address


    """

    name: Union[_UNDEFINED, PrefixSuffix, str] = UNDEFINED
    model_name_or_path: Union[_UNDEFINED, PrefixSuffix, str] = UNDEFINED

    def __post_init__(self):
        if self.model_name_or_path is UNDEFINED:
            assert isinstance(self.name, str)
            self.model_name_or_path = self.name


def copy_jsons_into_checkpoint_dir(one_dir_above: str, dirr: str):
    needed_files = [
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
    ]
    for f in needed_files:
        file_one_dir_above = f"{one_dir_above}/{f}"
        if not os.path.isfile(f"{dirr}/{f}"):
            assert os.path.isfile(file_one_dir_above), file_one_dir_above
            shutil.copy(file_one_dir_above, f"{dirr}/{f}")


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class DataArgs:
    """
    called DataTrainingArguments by huggingface: https://github.com/huggingface/transformers/blob/f0982682bd6fd0b438dda79ec45f3a8fac83a985/examples/research_projects/robust-speech-event/run_speech_recognition_ctc_bnb.py#L138
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # tilo: don't need these
    # max_train_samples: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
    #         "value if set."
    #     },
    # )
    # max_eval_samples: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
    #         "value if set."
    #     },
    # )
    eval_metrics: list[str] = list_field(
        default=["wer"],
        metadata={
            "help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "If :obj:`True`, will use the token generated when running"
            ":obj:`transformers-cli login` as HTTP bearer authorization for remote files."
        },
    )
    unk_token: str = field(
        default="<unk>",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="<pad>",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target language that should be used be"
            " passed to the tokenizer for tokenization. Note that"
            " this is only relevant if the model classifies the"
            " input audio to a sequence of phoneme sequences."
        },
    )


@dataclass
class ModelArgs(Buildable):
    """
    based on ModelArguments
    see: https://github.com/huggingface/transformers/blob/f0982682bd6fd0b438dda79ec45f3a8fac83a985/examples/research_projects/robust-speech-event/run_speech_recognition_ctc_bnb.py#L67
    """

    model_to_finetune: Union[_UNDEFINED, ModelIdentity] = UNDEFINED
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
        },
    )
    # tilos: cache_dir not needed here, see str(BASE_PATHES["transformers_cache_dir"])
    # cache_dir: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    # )
    text_cleaner: Union[TextCleaner, str] = UNDEFINED
    casing: Casing = Casing.upper

    new_vocab: Optional[list[str]] = None

    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    # --- dropouts ----
    # new-version: https://github.com/huggingface/transformers/blob/f0982682bd6fd0b438dda79ec45f3a8fac83a985/examples/research_projects/robust-speech-event/run_speech_recognition_ctc_bnb.py#L86
    # old-version: https://github.com/huggingface/transformers/blob/f0982682bd6fd0b438dda79ec45f3a8fac83a985/examples/research_projects/wav2vec2/run_common_voice.py#L63
    # tilo: changed from default 0.1 to 0.0
    # why did they change back to zero? no effect on WER? it has an effect on memory consumption, ~10% more on GPU during training
    #
    attention_dropout: Optional[float] = 0.0
    activation_dropout: Optional[float] = 0.0
    feat_proj_dropout: Optional[float] = 0.0
    hidden_dropout: Optional[float] = 0.0
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    # --- masking things ---
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    # tilos: this gradient_checkpointing was originall in training args!
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "The LayerDrop probability."}
    )
    ctc_loss_reduction: Optional[str] = field(
        default="mean",
        metadata={
            "help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."
        },
    )
    do_normalize_audio: bool = True  # TODO: WTF, this triggers multiple cache-dirs but does not really change the model!

    # --- non-init fields ---
    # just to stay closer to huggingface code
    model_name_or_path: str = field(init=False, repr=False)
    cache_dir: str = field(init=False, repr=False)

    # model_dir: Union[_UNDEFINED, str] = field(init=False, repr=True, default=None)
    # model: torch.nn.Module = field(init=False, repr=False, default=None)
    # processor: Wav2Vec2Processor = field(init=False, repr=False, default=None)
    # data_collator: DataCollatorCTCWithPadding = field(
    #     init=False, repr=False, default=UNDEFINED
    # )

    # _transcript_normalizer: Union[_UNDEFINED, TranscriptNormalizer] = field(
    #     init=False, default=UNDEFINED
    # )

    def __post_init__(self):
        self.model_name_or_path = self.model_to_finetune.model_name_or_path
        self.cache_dir = str(BASE_PATHES["transformers_cache_dir"])

    def _build_cache(self):
        raise NotImplementedError
        BASE_PATHES["transformers_cache_dir"] = PrefixSuffix(
            "base_path", "huggingface_cache/transformers"
        )

        if os.path.isdir(str(self.model_to_finetune.model_name_or_path)):
            self.model_dir = self.prefix_cache_dir("model")
            # TODO: does currently not work for pretrained models from hub
            shutil.copytree(
                str(self.model_to_finetune.model_name_or_path), self.model_dir
            )
            copy_jsons_into_checkpoint_dir(
                one_dir_above=f"{self.model_to_finetune.model_name_or_path}/..",
                dirr=self.model_dir,
            )
        else:
            # from huggingface hub or elsewhere
            self.model_dir = str(self.model_to_finetune.model_name_or_path)
            assert isinstance(
                self.model_dir, str
            ), f"not found on filesystem so should be huggingface-model"

    # def _post_build_setup(self) -> None:
    #     self.model_dir = read_json(self.dataclass_json)["model_dir"]
    #     self._build_processor_and_model()

    def _build_processor_and_model(self):
        raise NotImplementedError
        BASE_PATHES["transformers_cache_dir"] = PrefixSuffix(  # TODO: move this upwards
            "base_path", "huggingface_cache/transformers"
        )

        assert isinstance(self.model_dir, str)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_dir)
        self.processor.feature_extractor.do_normalize = self.do_normalize_audio
        assert isinstance(
            self.processor.current_processor, Wav2Vec2FeatureExtractor
        ), f"{type(self.processor.current_processor)=}"

        if self.new_vocab is not None:
            self._transcript_normalizer = TranscriptNormalizer(
                casing=self.casing,
                text_cleaner=self.text_cleaner_name,
                vocab=self.new_vocab,
            )

            new_encoder = {symbol: idx for idx, symbol in enumerate(self.new_vocab)}
            # if new_encoder != self.processor.tokenizer.encoder:
            original_model = Wav2Vec2ForCTC.from_pretrained(
                self.model_dir,
                cache_dir=str(BASE_PATHES["transformers_cache_dir"]),
                vocab_size=len(self.processor.tokenizer),
            )
            state_dict = original_model.state_dict()

            print(f"overwriting tokenizer with new vocab:{new_encoder}")
            vocab_json = self.prefix_cache_dir("new_vocab.json")
            with open(vocab_json, "w") as f:
                json.dump(new_encoder, f)

            """
            WTF! found this in huggingface code:
                    if self.do_lower_case:
                        text = text.upper()
            """
            # see. https://github.com/huggingface/transformers/issues/15333

            do_upper_case = self.casing is Casing.upper
            self.processor.tokenizer = Wav2Vec2CTCTokenizer(
                # see: https://github.com/huggingface/transformers/blob/692e61e91a0b83f5b847902ed619b7c74c0a5dda/examples/research_projects/wav2vec2/run_common_voice.py#L358
                vocab_file=vocab_json,
                pad_token="<pad>",
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                word_delimiter_token="|",
                do_lower_case=do_upper_case,  # yes its crazy, that lower-case means upper() in huggingface-world!
            )
            # remove head, cause vocab changed
            # see: https://discuss.huggingface.co/t/wav2vec2forctc-from-pretrained-for-already-trained-models/5716
            state_dict.pop("lm_head.weight")
            state_dict.pop("lm_head.bias")
        else:
            vocab = list(self.processor.tokenizer.get_vocab().keys())
            self._transcript_normalizer = TranscriptNormalizer(
                casing=self.casing, text_cleaner=self.text_cleaner_name, vocab=vocab
            )
            state_dict = None

        self.model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=self.model_to_finetune.model_name_or_path,
            state_dict=state_dict,
            cache_dir=str(BASE_PATHES["transformers_cache_dir"]),
            activation_dropout=self.activation_dropout,
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            feat_proj_dropout=self.feat_proj_dropout,
            mask_time_prob=self.mask_time_prob,
            # gradient_checkpointing=self.gradient_checkpointing, # transformers==4.18 does not know this?
            layerdrop=self.layerdrop,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
            ignore_mismatched_sizes=True,
        )
        if self.freeze_feature_extractor:
            self.model.freeze_feature_extractor()
        self.data_collator = DataCollatorCTCWithPadding(
            processor=self.processor, padding=True
        )

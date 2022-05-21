import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Union, Optional, Any

import torch.nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

from data_io.readwrite_files import read_json
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import (
    _UNDEFINED,
    UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from wav2vec2_finetuning.data_loading.data_collator import DataCollatorCTCWithPadding


@dataclass
class ModelIdentity:
    """
    formerly known as ModelToFinetune: name / handle / address


    """

    name: Union[_UNDEFINED, PrefixSuffix, str] = UNDEFINED
    model_name_or_path: Union[_UNDEFINED, PrefixSuffix, str] = UNDEFINED

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

@dataclass
class BaseModelForFinetuning(CachedData):
    """
    Arguments for training
    """

    model_to_finetune: Union[_UNDEFINED, ModelIdentity] = UNDEFINED
    text_normalizer: Union[_UNDEFINED, str] = UNDEFINED
    casing: Casing = Casing.upper

    new_vocab: Optional[list[str]] = None

    freeze_feature_extractor: Optional[bool] = True
    attention_dropout: Optional[float] = 0.1
    activation_dropout: Optional[float] = 0.1
    hidden_dropout: Optional[float] = 0.1
    feat_proj_dropout: Optional[float] = 0.1
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    layerdrop: Optional[float] = field(
        default=0.0, metadata={"help": "The LayerDrop probability."}
    )
    do_normalize_audio: bool = True  # TODO: WTF, this triggers multiple cache-dirs but does not really change the model!
    model_dir: Union[_UNDEFINED, str] = field(init=False, repr=True, default=None)

    model: torch.nn.Module = field(init=False, repr=False, default=None)
    processor: Wav2Vec2Processor = field(init=False, repr=False, default=None)
    data_collator: DataCollatorCTCWithPadding = field(
        init=False, repr=False, default=UNDEFINED
    )

    cache_base: PrefixSuffix = field(
        default_factory=lambda: BASE_PATHES["finetune_training"]
    )

    _transcript_normalizer: Union[_UNDEFINED, TranscriptNormalizer] = field(
        init=False, default=UNDEFINED
    )

    @property
    def name(self):
        return self.model_to_finetune.name

    # overriding is_ready should not be necessary
    # @property
    # def is_ready(self) -> bool:
    #     return self.data_collator is not UNDEFINED

    def _build_cache(self):
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
            self.model_dir = self.model_to_finetune.model_name_or_path
            assert isinstance(
                self.model_dir, str
            ), f"not found on filesystem so should be huggingface-model"

    def _post_build_setup(self) -> None:
        self.model_dir = read_json(self.dataclass_json)["model_dir"]
        self._build_processor_and_model()

    def _build_processor_and_model(self):
        """
        TODO: prepares model + processor for training?
        """
        BASE_PATHES["transformers_cache_dir"] = PrefixSuffix(
            "base_path", "huggingface_cache/transformers"
        )

        assert isinstance(self.model_dir, str)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_dir)
        self.processor.feature_extractor.do_normalize = self.do_normalize_audio

        if self.new_vocab is not None:
            self._transcript_normalizer = TranscriptNormalizer(
                casing=self.casing,
                text_normalizer=self.text_normalizer,
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
            vocab_json = "new_vocab.json"
            with open(vocab_json, "w") as f:
                json.dump(new_encoder, f)

            self.processor.tokenizer = Wav2Vec2CTCTokenizer(
                vocab_json,
                pad_token="<pad>",
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                word_delimiter_token="|",
            )
            # remove head, cause vocab changed
            raise NotImplementedError("TODO: still not working!")
            state_dict.pop("lm_head.weight")
            state_dict.pop("lm_head.bias")
        else:
            vocab = list(self.processor.tokenizer.get_vocab().keys())
            self._transcript_normalizer = TranscriptNormalizer(
                casing=self.casing, text_normalizer=self.text_normalizer, vocab=vocab
            )
            state_dict = None
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_to_finetune.model_name_or_path,
            state_dict=state_dict,
            cache_dir=str(BASE_PATHES["transformers_cache_dir"]),
            activation_dropout=self.activation_dropout,
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            feat_proj_dropout=self.feat_proj_dropout,
            mask_time_prob=self.mask_time_prob,
            gradient_checkpointing=self.gradient_checkpointing,
            layerdrop=self.layerdrop,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
        )
        if self.freeze_feature_extractor:
            self.model.freeze_feature_extractor()
        self.data_collator = DataCollatorCTCWithPadding(
            processor=self.processor, padding=True
        )

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Union, Any

import torch
from beartype import beartype

from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.models.wav2vec2.tokenization_wav2vec2 import (
    Wav2Vec2CTCTokenizerOutput,
)

from ctc_decoding.ctc_decoding import BaseCTCDecoder, AlignedBeams
from ctc_decoding.logit_aligned_transcript import LogitAlignedTranscript
from misc_utils.buildable import Buildable
from misc_utils.prefix_suffix import PrefixSuffix
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk


@dataclass
class HFCTCDecoder(BaseCTCDecoder, Buildable):

    tokenizer_name_or_path: Union[str, PrefixSuffix]
    _tokenizer: PreTrainedTokenizer = field(init=False)  # default=UNDEFINED ?

    def _build_self(self) -> Any:
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_name_or_path)
        )
        return self

    @property
    def vocab(self):
        return list(self._tokenizer.get_vocab().keys())

    @abstractmethod
    def decode(self, chunk: MessageChunk) -> AlignedBeams:
        raise NotImplementedError


@dataclass
class HFCTCGreedyDecoder(HFCTCDecoder):
    """
    huggingface does not have a "proper" greedy decoder, but does argmax somewhere in the asr-pipeline
    see: https://github.com/huggingface/transformers/blob/7999ec125fc31428ed6879bf01bb013483daf704/src/transformers/pipelines/automatic_speech_recognition.py#L323

    method called: convert_tokens_to_string in tokenization_wav2vec2
    see: https://github.com/huggingface/transformers/blob/7999ec125fc31428ed6879bf01bb013483daf704/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L254
    does ctc to text conversion (collapsing the sequence)
    """

    @beartype
    def decode(self, chunk: MessageChunk) -> AlignedBeams:

        greedy_path = torch.argmax(torch.from_numpy(chunk.array), dim=-1).squeeze()
        out: Wav2Vec2CTCTokenizerOutput = self._tokenizer.decode(  # noqa
            token_ids=greedy_path,
            output_char_offsets=True,
            skip_special_tokens=False,  # for ctc (see huggingface/transformers)
        )
        char_offsets: list[dict] = out.char_offsets
        vocab_space = [" "] + self.vocab
        vocab_space = [
            c for c in vocab_space if c not in ["<pad>", "<s>", "</s>", "<unk>", "|"]
        ]

        char_offsets = list(filter(lambda d: d["char"] in vocab_space, char_offsets))
        if len(char_offsets) == 0:
            char_offsets = [{"char": " ", "start_offset": 0}]

        return [
            LogitAlignedTranscript(
                text="".join([d["char"] for d in char_offsets]),
                logit_ids=[int(d["start_offset"]) for d in char_offsets],
            )
        ]

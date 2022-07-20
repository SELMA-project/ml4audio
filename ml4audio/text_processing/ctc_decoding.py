import itertools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

import torch
from beartype import beartype
from transformers import Wav2Vec2CTCTokenizer
from transformers.models.wav2vec2.tokenization_wav2vec2 import (
    Wav2Vec2CTCTokenizerOutput,
)

from misc_utils.beartypes import NumpyFloat2DArray, NeList, NeStr
from misc_utils.buildable import Buildable
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk

SeqVocIdx = tuple[int, int]

UNK = "<unk>"


@beartype
def strip_startend_idx(svi: list[SeqVocIdx], strip_idx: int) -> list[SeqVocIdx]:
    @beartype
    def pop_elements(
        svi: list[SeqVocIdx],
        strip_idx: int,
        do_reverse: bool = False,
    ) -> None:
        gen = reversed(list(enumerate(svi))) if do_reverse else enumerate(svi)
        for idx, (_seq_idx, voc_idx) in gen:
            if voc_idx == strip_idx:
                svi.pop(idx)
            else:
                break

    pop_elements(svi, strip_idx)
    pop_elements(svi, strip_idx, do_reverse=True)
    return svi


TokenSpans = list[tuple[str, tuple[int, int]]]


@beartype
def charwise_idx_for_tokenspans_via_linear_interpolation(
    token_spans: TokenSpans,
) -> list[int]:
    seq_idx = [
        round(start + (end - start) * k / len(word))  # interpolate
        for word, (start, end) in token_spans
        for k in range(len(word) + 1)
    ]
    return seq_idx[:-1]  # all but the last one, which is a space


@dataclass
class LogitAlignedTranscript:
    """
    Text is character-wise aligned to logits, no time-stamps here.
        logits == ctc-matrix
    """

    text: NeStr
    logit_ids: NeList[int]  # TODO: not too strict?

    logits_score: Optional[float] = None
    lm_score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate data."""
        have_same_len = len(self.text) == len(self.logit_ids)
        assert have_same_len, (
            f"{self.text=} and {self.logit_ids=} have different length! "
            + f"{len(self.text)=}!={len(self.logit_ids)=}"
        )

    @staticmethod
    def create_from_token_spans(
        token_spans: TokenSpans,
        lm_score: float,
        logits_score: float,
    ):
        text = " ".join([tok for tok, _ in token_spans])
        return LogitAlignedTranscript(
            text=text,
            logit_ids=charwise_idx_for_tokenspans_via_linear_interpolation(token_spans),
            lm_score=lm_score,
            logits_score=logits_score,
        )


@beartype
def get_prefix(
    idxs: list[int], blank_idx: int, silence_idx: int, vocab: list[str]
) -> list[tuple[int, str]]:
    """
    only used for greedy decoding -> TODO think about it
    :param idxs:
    """

    def replace_with_space(raw_letter: str) -> str:
        """Replace silence and UNK with space."""
        silence_str = vocab[silence_idx]
        if raw_letter in {silence_str, UNK}:
            letter = " "
        else:
            letter = raw_letter
        # TODO: wtf! why should the model ever predict unk?
        return letter

    svii = [
        (
            next(grp)[0],  # this is no int! stupid pycharm!
            vocab_idx,
        )
        for vocab_idx, grp in itertools.groupby(
            list(enumerate(idxs)),
            key=lambda xx: xx[1],
        )
    ]
    svii = [(g_idx, vocab_idx) for g_idx, vocab_idx in svii if vocab_idx != blank_idx]
    svii = strip_startend_idx(svii, strip_idx=silence_idx)
    # text = "".join([self.vocab[tt[1]] for tt in svii])
    # if UNK in text:
    #     # TODO
    #     pass

    return [
        (seq_idx, replace_with_space(vocab[vocab_idx])) for seq_idx, vocab_idx in svii
    ]


@beartype
def calc_silence_index(char2idx: dict[str, int]) -> int:
    if "<sep>" in char2idx:
        si = char2idx.get("<sep>")
    elif "|" in char2idx:
        si = char2idx.get("|")
    else:
        si = char2idx.get("</s>")
    return si


AlignedBeams = list[LogitAlignedTranscript]
BatchOfAlignedBeams = list[AlignedBeams]


@dataclass
class BaseCTCDecoder:
    @abstractmethod
    def decode(self, chunk: MessageChunk) -> AlignedBeams:
        raise NotImplementedError

    @beartype
    def decode_logits(self, logits: NumpyFloat2DArray) -> AlignedBeams:
        return self.decode(
            MessageChunk(message_id="foo", frame_idx=0, array=logits.squeeze())
        )


@dataclass
class HFCTCDecoder(BaseCTCDecoder, Buildable):

    tokenizer_name_or_path: str

    def _build_self(self) -> Any:
        self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            self.tokenizer_name_or_path
        )
        return self

    @property
    def vocab(self):
        return list(self._tokenizer.get_vocab().keys())

    @abstractmethod
    def decode(self, chunk: MessageChunk) -> AlignedBeams:
        raise NotImplementedError


NoneType = type(None)


@dataclass
class GreedyDecoder(HFCTCDecoder):
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
        bad_letters = [d["char"] for d in char_offsets if d["char"] not in vocab_space]

        if any((len(bad_letters) > 0 for bad_letters in bad_letters)):
            print(f"got bad letters: {bad_letters=}")
        char_offsets = list(filter(lambda d: d["char"] in vocab_space, char_offsets))
        if len(char_offsets) == 0:
            char_offsets = [{"char": " ", "start_offset": 0}]

        return [
            LogitAlignedTranscript(
                text="".join([d["char"] for d in char_offsets]),
                logit_ids=[int(d["start_offset"]) for d in char_offsets],
            )
        ]


def map_label(label: str) -> str:
    """
    TODO: not used anymore?
    Maps the wav2vec2 (characters) vocabulary to pyctcdecoders.

    :param label: some letter, character
    :return: character accepted by pyctcdecode
    """
    special_label_mapping = {"<pad>": "", "|": " "}
    map_funs = [
        lambda la: special_label_mapping.get(la, None),
        lambda la: "⁇" if len(la) > 1 else None,
    ]
    for map_fun in map_funs:
        out = map_fun(label)
        if out is not None:
            break
    else:
        out = label
    return out

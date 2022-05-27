import itertools
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from beartype import beartype
from misc_utils.beartypes import TorchTensor3D, TorchTensor2D
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED

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

    text: str
    logit_ids: list[int]

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
class BaseCTCDecoder(Buildable):
    """Abstract base-class for ctc-decoders."""

    lm_weight: Union[_UNDEFINED, float] = UNDEFINED
    beta: Union[_UNDEFINED, float] = UNDEFINED
    # cannot do this with beartype NeList[str] for vocab, cause it might be a CachedList
    vocab: Union[_UNDEFINED, list[str]] = UNDEFINED
    silence_idx: Optional[int] = None

    num_best: int = 1  # number of beams to return
    beam_size: int = 100

    def _build_self(self) -> None:
        self.vocab_size = len(self.vocab)
        assert self.vocab_size > 0
        char2idx = {char: idx for idx, char in enumerate(self.vocab)}

        self.blank_idx = char2idx.get("<pad>", char2idx.get("<s>"))
        self.silence_idx = calc_silence_index(char2idx)

    @abstractmethod
    def decode(self, ctc_matrix: TorchTensor2D) -> AlignedBeams:
        raise NotImplementedError

    def decode_batch(self, batched_ctc_matrix: TorchTensor3D) -> BatchOfAlignedBeams:
        batch_size, _seq_len, _vocab_size = batched_ctc_matrix.size()
        beams = [
            self.decode(batched_ctc_matrix[batch_idx].squeeze())
            for batch_idx in range(batch_size)
        ]
        return beams

    @property
    def silence_str(self) -> str:
        return self.vocab[self.silence_idx]

NoneType = type(None)

@dataclass
class GreedyDecoder(BaseCTCDecoder):

    lm_weight: NoneType = field(default=None, init=False, repr=False)
    beta: NoneType = field(default=None, init=False, repr=False)
    num_best: int = field(default=1, init=False, repr=False)
    beam_size: NoneType = field(default=None, init=False, repr=False)

    @beartype
    def decode(self, ctc_matrix: TorchTensor2D) -> AlignedBeams:
        greedy_path = torch.argmax(ctc_matrix, dim=-1)

        letters = get_prefix(
            idxs=greedy_path.tolist(),
            blank_idx=self.blank_idx,
            silence_idx=self.silence_idx,
            vocab=self.vocab,
        )
        return [
            LogitAlignedTranscript(
                text="".join([le for _, le in letters]),
                logit_ids=[idx for idx, _ in letters],
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
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from beartype import beartype
from transformers import (
    set_seed,
)

from misc_utils.beartypes import (
    NeList,
    NeStr,
    TorchTensor2D,
    NeNpFloatDim1,
    NeNpFloatDim1,
)
from misc_utils.buildable import Buildable
from ml4audio.text_processing.asr_text_cleaning import Casing, Letters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


def determine_casing(letter_vocab: Letters) -> Casing:
    more_than_half_is_upper = (
        sum([1 if c.upper() == c else 0 for c in letter_vocab]) > len(letter_vocab) / 2
    )
    casing = Casing.upper if more_than_half_is_upper else Casing.lower
    return casing


@dataclass
class ASRLogitsInferencer(Buildable):
    """
        Asr Connectionis temporal classification (CTC) Logits Inference

    ──────────────────────────────────────────────
    ──────│─────│───────│─────│───────│────────│──
    ──────│─────│───────│─────│───────│────────│──
    ──────│──┌───┬────┬───┐──┌┐───────│┌┐──────│──
    ──────│──│┌─┐│┌┐┌┐│┌─┐│──││───────┌┘└┐─────│──
    ──────│──││─└┴┘││└┤││└┘──││┌──┬──┬┼┐┌┼──┐──│──
    ──────│──││─┌┐─││─│││┌┬──┤││┌┐│┌┐├┤│││──┤──│──
    ──────│──│└─┘│─││─│└─┘├──┤└┤└┘│└┘│││└┼──│──│──
    ──────│──└───┘─└┘─└───┘──└─┴──┴─┐├┘└─┴──┘──│──
    ──────│─────│───────│─────│───┌─┘││────────│──
    ──────│─────│───────│─────│───└──┘│────────│──
    ──────│─────│───────│─────│───────│────────│──
    ──────│─────│───────│─────│───────│────────│──

    """

    asr_model_sample_rate: ClassVar[int] = 16000

    @property
    @beartype
    def name(self) -> NeStr:
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab(self) -> NeList[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def letter_vocab(self) -> Letters:
        raise NotImplementedError

    @abstractmethod
    @beartype
    def calc_logits(self, audio: NeNpFloatDim1) -> TorchTensor2D:
        raise NotImplementedError

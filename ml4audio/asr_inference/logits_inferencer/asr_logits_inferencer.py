from abc import abstractmethod
from dataclasses import dataclass
from typing import Union, Annotated, ClassVar

import torch
from beartype import beartype
from beartype.vale import IsAttr, IsEqual
from misc_utils.beartypes import (
    NumpyFloat1DArray,
    NeList,
    NeStr,
    TorchTensor2D,
)
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import (
    UNDEFINED,
)
from numpy import floating, int16
from numpy.typing import NDArray
from transformers import (
    set_seed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NumpyFloatORInt16_1DArray = Annotated[
    Union[NDArray[floating], NDArray[int16]], IsAttr["ndim", IsEqual[1]]
]

set_seed(42)


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

    @abstractmethod
    @beartype
    def calc_logits(self, audio: NumpyFloat1DArray) -> TorchTensor2D:
        raise NotImplementedError

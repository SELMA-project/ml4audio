from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Iterable,
    Iterator,
)

from misc_utils.beartypes import (
    NumpyFloat1DArray,
    NeStr,
)

ArrayText = tuple[NumpyFloat1DArray, NeStr]


IdArray = tuple[NeStr, NumpyFloat1DArray]
IdText = tuple[NeStr, NeStr]


@dataclass
class AudioData(Iterable[IdArray]):
    sample_rate: int

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[IdArray]:
        raise NotImplementedError


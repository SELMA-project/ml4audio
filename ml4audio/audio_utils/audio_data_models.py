from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Iterable,
    Iterator, Any, Union,
)

from misc_utils.beartypes import (
    NumpyFloat1DArray,
    NeStr,
)
from misc_utils.dataclass_utils import FillUndefined, _UNDEFINED, UNDEFINED

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


@dataclass
class AudioTextData(Iterable[ArrayText]):
    """
    naming: Auteda == Audio Text Data
    """

    sample_rate: int

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[ArrayText]:
        raise NotImplementedError

@dataclass
class FileLikeAudioDatum:
    id: str
    audio_source: Any  # BytesIO, ExFileObject
    format: str

@dataclass
class FileLikeAudioCorpus(Iterable[FileLikeAudioDatum], FillUndefined):
    # TODO: why is this not buildable?
    id: Union[_UNDEFINED, NeStr] = UNDEFINED

    @abstractmethod
    def __iter__(self) -> Iterator[FileLikeAudioDatum]:
        raise NotImplementedError

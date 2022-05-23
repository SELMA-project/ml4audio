from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Iterable,
    Iterator,
    Any,
    Union,
    Optional,
)

from misc_utils.beartypes import (
    NumpyFloat1DArray,
    NeStr,
)
from misc_utils.dataclass_utils import FillUndefined, _UNDEFINED, UNDEFINED
from misc_utils.utils import Singleton

ArrayText = tuple[NumpyFloat1DArray, NeStr]


IdArray = tuple[NeStr, NumpyFloat1DArray]
IdText = tuple[NeStr, NeStr]


@dataclass
class StableDatum:
    """
    pure semantic class just to remind that implementation serves as interface, should ideally not change
    """

    pass


@dataclass
class FileLikeAudioDatum(StableDatum):
    id: str
    audio_source: Any  # BytesIO, ExFileObject
    format: str


@dataclass
class AudioFile(FileLikeAudioDatum):

    format: Optional[str] = field(init=False, repr=False)

    def __post_init__(self):
        self.format = self.audio_source.split(".")[-1]


@dataclass
class TranscriptAnnotation(StableDatum):
    segment_id: str
    text: str


@dataclass
class _UNKNOWN_START_END(metaclass=Singleton):
    pass


UNKNOWN_START_END = _UNKNOWN_START_END()


@dataclass
class AlignmentSpan:
    start_a: Union[int, float, _UNKNOWN_START_END]
    end_a: Union[int, float, _UNKNOWN_START_END]
    start_b: Union[int, float, _UNKNOWN_START_END]
    end_b: Union[int, float, _UNKNOWN_START_END]


@dataclass
class AlignmentSpanAnnotation(AlignmentSpan):
    confidence: float


@dataclass
class StandAloneAlignmentSpanAnnotation(AlignmentSpanAnnotation):
    id_seq_a: str
    id_seq_b: str


@dataclass
class SequenceAlignment:
    id_seq_a: str
    id_seq_b: str
    alignments: list[AlignmentSpan]


@dataclass
class AlignmentCorpus(Iterable[SequenceAlignment], FillUndefined):
    id_corpus_a: Union[_UNDEFINED, NeStr] = UNDEFINED
    id_corpus_b: Union[_UNDEFINED, NeStr] = UNDEFINED

    @abstractmethod
    def __iter__(self) -> Iterator[SequenceAlignment]:
        raise NotImplementedError


@dataclass
class SegmentAnnotation(StableDatum):
    id: str
    audio_id: str
    start: float = 0.0  # in sec
    end: Optional[float] = None  # TODO!!!

    @property
    def duration(self) -> Optional[float]:
        """
        in seconds
        """
        if self.end is not None:
            return self.end - self.start
        else:
            return None


@dataclass
class AudioFileCorpus(Iterable[AudioFile], FillUndefined):
    id: Union[_UNDEFINED, NeStr] = UNDEFINED

    @abstractmethod
    def __iter__(self) -> Iterator[AudioFile]:
        raise NotImplementedError


@dataclass
class SegmentCorpus(Iterable[SegmentAnnotation], FillUndefined):
    id: Union[_UNDEFINED, NeStr] = UNDEFINED
    audiocorpus_id: Union[_UNDEFINED, NeStr] = UNDEFINED

    @abstractmethod
    def __iter__(self) -> Iterator[SegmentAnnotation]:
        raise NotImplementedError


@dataclass
class TranscriptCorpus(Iterable[TranscriptAnnotation], FillUndefined):
    """
    this serves as Interface
    """

    id: Union[_UNDEFINED, NeStr] = UNDEFINED
    segmentcorpus_id: Union[_UNDEFINED, NeStr] = UNDEFINED

    @abstractmethod
    def __iter__(self) -> Iterator[TranscriptAnnotation]:
        raise NotImplementedError


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
class FileLikeAudioCorpus(Iterable[FileLikeAudioDatum], FillUndefined):
    # TODO: why is this not buildable?
    id: Union[_UNDEFINED, NeStr] = UNDEFINED

    @abstractmethod
    def __iter__(self) -> Iterator[FileLikeAudioDatum]:
        raise NotImplementedError

from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated

from beartype.vale import Is

from misc_utils.beartypes import NumpyFloat1D
from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEndArray,
    StartEndText,
    is_non_overlapping,
)


class SetupTearDown:
    @abstractmethod
    def __enter__(self):
        """
        use to load the model into memory, prepare things
        """
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """
        use as tear-down, to free memory, unload model
        """
        raise NotImplementedError


StartEndTextsNonOverlap = Annotated[
    list[StartEndText],
    Is[is_non_overlapping],
]


@dataclass
class AudioArray2SegmentedTranscripts(SetupTearDown):
    """
    TODO: AA2ST = Audio Array 2 Segmented Transcripts
    """
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        return 16000

    @abstractmethod
    def audio_to_segmented_transcripts(
        self, audio_array: NumpyFloat1D
    ) -> StartEndTextsNonOverlap:
        raise NotImplementedError

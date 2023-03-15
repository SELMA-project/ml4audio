from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np

from misc_utils.beartypes import NumpyFloat1D
from misc_utils.buildable import Buildable
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript


class SetupTearDown:
    @abstractmethod
    def __enter__(self):
        """
        use to load the model into memory, prepare things
        """
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        use as tear-down, to free memory, unload model
        """
        raise NotImplementedError


@dataclass
class ASRAudioArrayInferencer(SetupTearDown):
    @property
    def sample_rate(self) -> int:
        return 16000

    @abstractmethod
    def transcribe_audio_array(self, audio_array: NumpyFloat1D) -> str:
        raise NotImplementedError

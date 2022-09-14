from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np

from misc_utils.buildable import Buildable
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript


@dataclass
class ASRAudioArrayInferencer(Buildable, ABC):
    """
    TODO: why buildable here?
    """

    @abstractmethod
    def transcribe_audio_array(self, audio_array: np.ndarray) -> AlignedTranscript:
        raise NotImplementedError

    def __enter__(self):
        return self.build()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

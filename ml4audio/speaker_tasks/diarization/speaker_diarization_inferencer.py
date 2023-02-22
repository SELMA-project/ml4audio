from dataclasses import dataclass

from beartype import beartype
from misc_utils.buildable import Buildable

from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEndArraysNonOverlap,
    StartEndLabels,
)


@dataclass
class SpeakerDiarizationInferencer(Buildable):
    @beartype
    def predict(self, s_e_a: StartEndArraysNonOverlap) -> StartEndLabels:
        raise NotImplementedError

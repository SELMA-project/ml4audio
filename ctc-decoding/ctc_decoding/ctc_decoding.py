from abc import abstractmethod
from dataclasses import dataclass

from ctc_decoding.logit_aligned_transcript import LogitAlignedTranscript
from misc_utils.beartypes import NumpyFloat2DArray

NoneType = type(None)

AlignedBeams = list[LogitAlignedTranscript]
BatchOfAlignedBeams = list[AlignedBeams]


@dataclass
class BaseCTCDecoder:
    @abstractmethod
    def ctc_decode(self, logits: NumpyFloat2DArray) -> AlignedBeams:
        raise NotImplementedError

from abc import abstractmethod
from dataclasses import dataclass

from beartype import beartype

from ctc_decoding.logit_aligned_transcript import LogitAlignedTranscript
from misc_utils.beartypes import NumpyFloat2DArray
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk

NoneType = type(None)

AlignedBeams = list[LogitAlignedTranscript]
BatchOfAlignedBeams = list[AlignedBeams]


@dataclass
class BaseCTCDecoder:
    @abstractmethod
    def decode(self, chunk: MessageChunk) -> AlignedBeams:
        raise NotImplementedError

    @beartype
    def decode_logits(self, logits: NumpyFloat2DArray) -> AlignedBeams:
        return self.decode(
            MessageChunk(message_id="foo", frame_idx=0, array=logits.squeeze())
        )

from dataclasses import dataclass
from typing import Iterator, Iterable

from beartype import beartype

from ml4audio.asr_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from ml4audio.asr_inference.transcript_glueing import NonEmptyAlignedTranscript
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript
from ml4audio.audio_utils.overlap_array_chunker import AudioMessageChunk


@dataclass
class ASRMessage:
    message_id: str
    aligned_transcript: NonEmptyAlignedTranscript  # TODO: enforce being non-empty?
    end_of_message: bool


@beartype
def transcribe_stream(
    inferencer: HFASRDecodeInferencer, input_it: Iterable[AudioMessageChunk]
) -> Iterator[ASRMessage]:
    for datum in input_it:
        # print(f"{datum=}")
        tr: AlignedTranscript = inferencer.transcribe_audio_array(datum.array)
        tr.set_abs_pos_in_time(datum.frame_idx)
        yield ASRMessage(
            message_id=datum.message_id,
            aligned_transcript=tr,
            end_of_message=datum.end_of_signal,
        )

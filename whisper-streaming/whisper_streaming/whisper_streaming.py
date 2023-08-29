from dataclasses import dataclass, field
from typing import Iterator, Optional, Any, ClassVar

from beartype import beartype
from transformers import set_seed

from ctc_asr_chunked_inference.asr_infer_decode import (
    convert_and_resample,
)
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import (
    UNDEFINED,
)
from ml4audio.asr_inference.faster_whisper_inferencer import (
    FasterWhisperASRSegmentInferencer,
)
from ml4audio.asr_inference.inference import StartEndTextsNonOverlap, SetupTearDown
from ml4audio.audio_utils.overlap_array_chunker import (
    OverlapArrayChunker,
    AudioMessageChunk,
    MessageChunk,
)
from whisper.audio import SAMPLE_RATE as WHISPER_SAMPLE_RATE

set_seed(42)


@dataclass
class WhisperStreamer(Buildable, SetupTearDown):
    """

    does:
    1. chunking
    2. asr-inference = inference + decoding
    3. transcript updating

    """

    input_sample_rate: int = 16_000
    asr_inferencer: FasterWhisperASRSegmentInferencer = UNDEFINED

    audio_bufferer: Optional[OverlapArrayChunker] = field(
        init=True, repr=True, default=None
    )

    model_sample_rate: ClassVar[int] = WHISPER_SAMPLE_RATE

    def reset(self) -> None:
        self.audio_bufferer.reset()

    @property
    def name(self):
        return f"streaming-{self.asr_inferencer.name}"

    def _build_self(self) -> Any:
        self.reset()

    def __enter__(self):
        self.asr_inferencer.__enter__()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.asr_inferencer.__exit__(exc_type, exc_val, exc_tb)

    @beartype
    def handle_inference_input(
        self, inpt: AudioMessageChunk
    ) -> list[StartEndTextsNonOverlap]:
        segments: list[StartEndTextsNonOverlap] = []
        for chunk in self.audio_bufferer.handle_datum(inpt):
            chunk: MessageChunk
            audio_array = convert_and_resample(
                chunk.array,
                self.input_sample_rate,
                self.model_sample_rate,
            )

            this_chunks_transcript_segments = (
                self.asr_inferencer.predict_transcribed_with_whisper_args(
                    audio_array, self.asr_inferencer.whisper_args
                )
            )
            chunk_offset = (chunk.frame_idx) / self.input_sample_rate
            this_chunks_transcript_segments = [
                (s + chunk_offset, e + chunk_offset, t)
                for s, e, t in this_chunks_transcript_segments
            ]
            segments.append(this_chunks_transcript_segments)
        return segments

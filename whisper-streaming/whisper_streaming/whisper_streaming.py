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


def concat_transcript(segments: StartEndTextsNonOverlap) -> str:
    return "".join([t for _, _, t in segments])


@dataclass
class OverlappingSegment:
    """
    has no start, cause it is somehow overlapping with the previous segment!
    """

    end: float
    append_suffix: str
    remove_suffix: Optional[str] = None


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
    max_prompt_len_int_letters: int = 100
    audio_bufferer: Optional[OverlapArrayChunker] = field(
        init=True, repr=True, default=None
    )
    transcripts_buffer: Optional[StartEndTextsNonOverlap] = field(
        init=True, repr=False, default_factory=lambda: []
    )

    model_sample_rate: ClassVar[int] = WHISPER_SAMPLE_RATE

    def reset(self) -> None:
        self.audio_bufferer.reset()
        self.transcripts_buffer = []

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
    ) -> Iterator[tuple[OverlappingSegment, StartEndTextsNonOverlap]]:
        for chunk in self.audio_bufferer.handle_datum(inpt):
            # print(f"chunk-dur: {len(chunk.array)/self.input_sample_rate}")
            overlap_segment, non_overlapping_segments = self._transcribe_chunk(
                chunk, self.transcripts_buffer
            )

            smaller_than_and_non_negative = 0.0
            self.transcripts_buffer = [
                (
                    smaller_than_and_non_negative,
                    overlap_segment.end,
                    overlap_segment.append_suffix,
                )
            ] + non_overlapping_segments
            yield overlap_segment, non_overlapping_segments

    def _transcribe_chunk(
        self, chunk: MessageChunk, transcripts_buffer: StartEndTextsNonOverlap
    ) -> tuple[OverlappingSegment, StartEndTextsNonOverlap]:

        audio_array = convert_and_resample(
            chunk.array,
            self.input_sample_rate,
            self.model_sample_rate,
        )
        chunk_offset = float(chunk.frame_idx) / self.input_sample_rate
        whisper_args = self.asr_inferencer.whisper_args

        remove_suffix, whisper_args.prefix = self._whisperprefix_and_removesuffix(
            chunk_offset, transcripts_buffer
        )
        this_chunks_transcript_segments = [
            (s + chunk_offset, e + chunk_offset, t)
            for s, e, t in self.asr_inferencer.predict_transcribed_with_whisper_args(
                audio_array, whisper_args
            )
        ]
        assert (
            this_chunks_transcript_segments[0][0] == chunk_offset
        )  # this actually doesn't matter

        (
            _first_start_is_invalid,
            first_end,
            first_transcript,
        ) = this_chunks_transcript_segments[0]
        return (
            OverlappingSegment(
                end=first_end,
                append_suffix=first_transcript,
                remove_suffix=remove_suffix,
            ),
            this_chunks_transcript_segments[1:],
        )

    def _whisperprefix_and_removesuffix(
        self, chunk_offset: float, transcripts_buffer: StartEndTextsNonOverlap
    ) -> tuple[Optional[str], Optional[str]]:
        prefix_from, prefix_to = -4, -1  # TODO: which values here?
        segments_within_this_chunk = [
            (s, e, t) for s, e, t in transcripts_buffer if e >= chunk_offset
        ]
        if len(segments_within_this_chunk) > 0:
            words_inside_this_chunk = concat_transcript(
                segments_within_this_chunk
            ).split(" ")
            whisper_prefix = " ".join(words_inside_this_chunk[prefix_from:prefix_to])
            remove_suffix = " ".join(words_inside_this_chunk[prefix_from:])
        else:
            whisper_prefix = None
            remove_suffix = None
        return remove_suffix, whisper_prefix

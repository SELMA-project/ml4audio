"""
1. buffer audio-arrays -> buffering
2. transcribe -> stateful but NOT buffering
3. glue transcripts -> buffering
"""
from dataclasses import dataclass, field
from typing import Iterator, Optional, Iterable, Annotated, Any

import numpy as np
from beartype import beartype
from beartype.vale import Is
from transformers import set_seed

from ctc_asr_chunked_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from misc_utils.beartypes import Numpy1D, NumpyInt16_1D, NeList
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import (
    UNDEFINED,
)
from ml4audio.asr_inference.transcript_glueing import (
    NO_NEW_SUFFIX,
    accumulate_transcript_suffixes,
)
from ml4audio.asr_inference.transcript_gluer import (
    TranscriptGluer,
    ASRStreamInferenceOutput,
)
from ml4audio.audio_utils.aligned_transcript import TimestampedLetters
from ml4audio.audio_utils.audio_io import (
    convert_to_16bit_array,
    break_array_into_chunks,
)
from ml4audio.audio_utils.overlap_array_chunker import (
    OverlapArrayChunker,
    AudioMessageChunk,
    audio_messages_from_chunks,
)

set_seed(42)


@dataclass
class Aschinglupi(Buildable):
    """

        naming: Aschinglupi == ASR Chunking Inference Gluing Pipeline
        does:
        1. chunking
        2. asr-inference
        3. transcript glueing
        TODO: split it into pieces!

    ─────▀▀▌───────▐▀▀
    ─────▄▀░◌░░░░░░░▀▄
    ────▐░░◌░▄▀██▄█░░░▌
    ────▐░░░▀████▀▄░░░▌
    ────═▀▄▄▄▄▄▄▄▄▄▄▄▀═

    """

    hf_asr_decoding_inferencer: HFASRDecodeInferencer = UNDEFINED
    transcript_gluer: TranscriptGluer = UNDEFINED
    # step_dur: Union[
    #     float, int
    # ] = 1.0  # union cause json (de)serialization sometimes loses float information!
    # window_dur: Union[float, int] = 4.0
    # dtype: str = "int16"
    audio_bufferer: Optional[OverlapArrayChunker] = field(
        init=True, repr=True, default=None
    )

    def reset(self) -> None:
        self.audio_bufferer.reset()
        self.transcript_gluer.reset()

    @property
    def sample_rate(self) -> int:
        return self.hf_asr_decoding_inferencer.input_sample_rate

    @property
    def name(self):
        # TODO: why is decoder not part of name?
        return f"aschinglupi-{self.hf_asr_decoding_inferencer.logits_inferencer.name}"

    def _build_self(self) -> Any:
        # self.hf_asr_decoding_inferencer.build()  # TODO(tilo): WTF!! there should be no build in a _post_build_setup!
        # TODO: might be handled by proper is_ready logic!
        assert self.hf_asr_decoding_inferencer._was_built
        assert self.hf_asr_decoding_inferencer.logits_inferencer._was_built

        # self.audio_bufferer.reset()
        # self.transcript_gluer.build()  # this is somewhat annoying, that this buildable-object is not getting build cause it is child of cacheddata
        assert self.transcript_gluer.seqmatcher is not None
        self.reset()

    @beartype
    def handle_inference_input(
        self, inpt: AudioMessageChunk
    ) -> Iterator[ASRStreamInferenceOutput]:
        for chunk in self.audio_bufferer.handle_datum(inpt):
            chunk: AudioMessageChunk
            letters = self.hf_asr_decoding_inferencer.transcribe_audio_array(
                chunk.array
            )
            letters.timestamps += (chunk.frame_idx) / self.sample_rate
            new_suffix = self.transcript_gluer.calc_transcript_suffix(letters)
            if new_suffix is not NO_NEW_SUFFIX:
                yield ASRStreamInferenceOutput(
                    id=chunk.message_id,
                    aligned_transcript=new_suffix,
                    end_of_message=chunk.end_of_signal,
                )

    @property
    def vocab(self) -> list[str]:
        return self.hf_asr_decoding_inferencer.vocab


@beartype
def is_end_of_signal(am: AudioMessageChunk) -> bool:
    return am.end_of_signal


CompleteMessage = Annotated[
    NeList[AudioMessageChunk], Is[lambda ams: is_end_of_signal(ams[-1])]
]

NO_TRANSCRIPT = " ... "


@beartype
def aschinglupi_transcribe_chunks(
    inferencer: Aschinglupi, chunks: Iterable[NumpyInt16_1D]
) -> TimestampedLetters:
    audio_messages = list(
        audio_messages_from_chunks(
            signal_id="nobody_cares",
            chunks=chunks,
        )
    )
    inferencer.reset()

    outputs: list[ASRStreamInferenceOutput] = [
        t for inpt in audio_messages for t in inferencer.handle_inference_input(inpt)
    ]
    suffixes_g = (tr.aligned_transcript for tr in outputs)
    transcript = accumulate_transcript_suffixes(suffixes_g)
    inferencer.reset()
    return transcript


@beartype
def transcribe_audio_array(
    inferencer: Aschinglupi, array: Numpy1D, chunk_dur: float = 4.0
) -> TimestampedLetters:
    if array.dtype is not np.int16:
        array = convert_to_16bit_array(array)
    chunks = break_array_into_chunks(array, int(inferencer.sample_rate * chunk_dur))
    last_response = aschinglupi_transcribe_chunks(inferencer, chunks)
    return last_response


"""
if __name__ == "__main__":
    die_if_unbearable(
        [
            AudioMessageChunk(
                "foo", 0, np.zeros((9,), dtype=np.int16), end_of_signal=True
            )
        ],
        CompleteMessage,
    )
"""

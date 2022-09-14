"""
1. buffer audio-arrays -> buffering
2. transcribe -> stateful but NOT buffering
3. glue transcripts -> buffering
"""
from dataclasses import dataclass, field
from typing import Iterator, Union, Optional, Iterable

from beartype import beartype
from transformers import set_seed

from ml4audio.asr_inference.asr_array_stream_inference import ASRMessage
from ml4audio.asr_inference.hfwav2vec2_asr_decode_inferencer import \
    HFASRDecodeInferencer
from ml4audio.asr_inference.transcript_gluer import TranscriptGluer, \
    ASRStreamInferenceOutput
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import (
    UNDEFINED,
    _UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.audio_utils.overlap_array_chunker import (
    OverlapArrayChunker,
    AudioMessageChunk,
)

set_seed(42)


@dataclass
class Aschinglupi(CachedData):
    """

        nameing: Aschinglupi == ASR Chunking Inference Glueing Pipeline
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

    hf_asr_decoding_inferencer: Union[_UNDEFINED, HFASRDecodeInferencer] = UNDEFINED
    transcript_gluer: Union[_UNDEFINED, TranscriptGluer] = UNDEFINED
    # step_dur: Union[
    #     float, int
    # ] = 1.0  # union cause json (de)serialization sometimes loses float information!
    # window_dur: Union[float, int] = 4.0
    # dtype: str = "int16"
    audio_bufferer: Optional[OverlapArrayChunker] = field(
        init=True, repr=True, default=None
    )

    cache_base: PrefixSuffix = field(
        default_factory=lambda: BASE_PATHES["asr_inference"]
    )

    def reset(self) -> None:
        self.audio_bufferer.reset()
        self.transcript_gluer.reset()

    @property
    def name(self):
        # TODO: why is decoder not part of name?
        return f"aschinglupi-{self.hf_asr_decoding_inferencer.logits_inferencer.name}"

    def _build_cache(self):
        """
        it is bundling/glueing things together does not have own cache
        usefull for exporting this into an asr-service
        """
        pass

    def _post_build_setup(self):
        """
        this object is CachedData just for documentation purposes
        thats why it still needs to build its children after loading from cache

        """

        self.hf_asr_decoding_inferencer.build()  # TODO(tilo): WTF!! there should be no build in a _post_build_setup!
        # TODO: might be handled by proper is_ready logic!
        assert self.hf_asr_decoding_inferencer._was_built
        assert self.hf_asr_decoding_inferencer.logits_inferencer._was_built

        self.audio_bufferer.reset()
        self.transcript_gluer.build()  # this is somewhat annoying, that this buildable-object is not getting build cause it is child of cacheddata
        assert self.transcript_gluer.seqmatcher is not None

    @beartype
    def handle_inference_input(
        self, inpt: AudioMessageChunk
    ) -> Iterator[ASRStreamInferenceOutput]:
        for chunk in self.audio_bufferer.handle_datum(inpt):
            chunk: AudioMessageChunk
            altr: AlignedTranscript = (
                self.hf_asr_decoding_inferencer.transcribe_audio_array(chunk.array)
            )
            altr.frame_id = chunk.frame_idx
            altr.update_offset(chunk.frame_idx)
            message = ASRMessage(
                message_id=chunk.message_id,
                aligned_transcript=altr,
                end_of_message=chunk.end_of_signal,
            )
            glued_transcript = self.transcript_gluer.handle_message(message)
            yield glued_transcript

    # @beartype
    # async def _async_transcribe_audio_stream(
    #     self, buffered_audio_g: AsyncIterable[AudioChunk]
    # ) -> AsyncIterator[ASRMessage]:
    #     async for datum in buffered_audio_g:
    #         tr: AlignedTranscript = (
    #             self.hf_asr_decoding_inferencer.transcribe_audio_array(
    #                 datum.audio_array
    #             )
    #         )
    #         tr.set_abs_pos_in_time(datum.frame_idx)
    #         yield ASRMessage(
    #             message_id=datum.message_id,
    #             aligned_transcript=tr,
    #             end_of_message=datum.end_of_signal,
    #         )

    @property
    def vocab(self) -> list[str]:
        return self.hf_asr_decoding_inferencer.vocab


@beartype
def gather_final_aligned_transcripts(
    aschinglupi: Aschinglupi, asr_input: Iterable[AudioMessageChunk]
) -> Iterator[ASRStreamInferenceOutput]:
    """
    gather/accumulate intermediate transcripts until final is received
    """
    last_response: ASRStreamInferenceOutput = None

    for inpt in asr_input:
        for t in aschinglupi.handle_inference_input(inpt):
            last_response = t

        if inpt.end_of_signal:
            aschinglupi.reset()
            yield last_response
import os
from time import time
from typing import Optional

import numpy as np
import pytest

from conftest import get_test_cache_base
from ctc_asr_chunked_inference.asr_chunk_infer_glue_pipeline import Aschinglupi
from ctc_asr_chunked_inference.asr_infer_decode import ASRInferDecoder
from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.asr_inference.transcript_glueing import (
    accumulate_transcript_suffixes,
)
from ml4audio.asr_inference.transcript_gluer import (
    TranscriptGluer,
    ASRStreamInferenceOutput,
)
from ml4audio.audio_utils.overlap_array_chunker import (
    audio_messages_from_file,
    OverlapArrayChunker,
)
from ml4audio.text_processing.asr_metrics import calc_cer
from ml4audio.text_processing.asr_text_normalization import (
    normalize_filter_text,
    Casing,
)
from ml4audio.text_processing.pretty_diff import smithwaterman_aligned_icdiff

BASE_PATHES["asr_inference"] = get_test_cache_base()
os.environ["DEBUG_GLUER"] = "True"


@pytest.mark.parametrize(
    "step_dur,window_dur,max_step_dur,chunk_dur,max_CER,num_responses",
    [
        # fmt: off
        (1.0, 2.0, None,0.1, 0.073, 25),
        # got worse due to using opus instead of wav
        (1.5, 3.0, None,0.1, 0.016, 17),
        (1.0, 4.0, None,0.1, 0.008, 25),

        (2.0, 4.0, None, 0.1, 0.0053, 13),
        (1.0, 4.0, 2.0, 2.0, 0.0053, 13), # same as above cause max_step_dur == chunk_dur == 2.0, the min_step_dur is kind of ignored, cause chunk_dur is fixed

        (4.0, 8.0, None,0.1, 0.0027, 7),
        (1.0, 8.0, None,0.1, 0.0, 25),
        # fmt: on
    ],
)
def test_ASRStreamInferencer(
    asr_infer_decoder: ASRInferDecoder,
    librispeech_audio_file,
    librispeech_ref,
    step_dur: float,
    window_dur: float,
    max_step_dur: Optional[float],
    chunk_dur: float,
    max_CER: float,
    num_responses: int,
):

    SR = expected_sample_rate = asr_infer_decoder.input_sample_rate
    asr_input = list(
        audio_messages_from_file(
            librispeech_audio_file, expected_sample_rate, chunk_duration=chunk_dur
        )
    )
    assert asr_input[-1].end_of_signal
    audio_signal = np.concatenate([ac.array for ac in asr_input])
    wav_length = 393920
    opus_is_alittle_longer = 70
    assert audio_signal.shape[0] == wav_length + opus_is_alittle_longer
    # audio_duration = audio_signal.shape[0] / SR

    streaming_asr: Aschinglupi = Aschinglupi(
        hf_asr_decoding_inferencer=asr_infer_decoder,
        transcript_gluer=TranscriptGluer(),
        audio_bufferer=OverlapArrayChunker(
            chunk_size=int(window_dur * SR),
            minimum_chunk_size=int(1 * SR),  # one second!
            min_step_size=int(step_dur * SR),
            max_step_size=int(max_step_dur * SR) if max_step_dur is not None else None,
        ),
    ).build()

    streaming_asr.reset()

    outputs: list[ASRStreamInferenceOutput] = [
        t for inpt in asr_input for t in streaming_asr.handle_inference_input(inpt)
    ]
    assert len(outputs) == num_responses
    assert outputs[-1].end_of_message

    suffixes_g = (tr.aligned_transcript for tr in outputs)
    transcript = accumulate_transcript_suffixes(suffixes_g)
    hyp = transcript.letters.strip(" ")

    # print(f"{audio_duration,prefix.timestamps[-1]}")
    ref = normalize_filter_text(
        librispeech_ref,
        asr_infer_decoder.logits_inferencer.vocab,
        text_cleaner="en",
        casing=Casing.upper,
    )
    # print(smithwaterman_aligned_icdiff(ref, hyp))
    cer = calc_cer([ref], [hyp])
    print(f"{step_dur=},{window_dur=},{cer=}")

    assert cer <= max_CER

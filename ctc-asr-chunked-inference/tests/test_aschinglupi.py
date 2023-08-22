import os

import numpy as np
import pytest
from time import time

from conftest import get_test_cache_base
from ctc_asr_chunked_inference.asr_chunk_infer_glue_pipeline import Aschinglupi
from ctc_asr_chunked_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.asr_inference.transcript_glueing import remove_and_append
from ml4audio.asr_inference.transcript_gluer import (
    TranscriptGluer,
    ASRStreamInferenceOutput,
)
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript
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
    "step_dur,window_dur,max_CER,num_responses",
    [
        (1.0, 2.0, 0.085, 25),  # got worse due to using opus instead of wav
        (1.5, 3.0, 0.02, 17),
        (1.0, 4.0, 0.0098, 25),
        (2.0, 4.0, 0.0065, 13),
        (4.0, 8.0, 0.0033, 7),
        (1.0, 8.0, 0.0, 25),
    ],
)
def test_ASRStreamInferencer(
    asr_decode_inferencer: HFASRDecodeInferencer,
    librispeech_audio_file,
    librispeech_raw_ref,
    step_dur: float,
    window_dur: float,
    max_CER: float,
    num_responses: int,
):

    SR = (
        expected_sample_rate
    ) = asr_decode_inferencer.logits_inferencer.input_sample_rate
    asr_input = list(
        audio_messages_from_file(librispeech_audio_file, expected_sample_rate)
    )
    assert asr_input[-1].end_of_signal
    audio_signal = np.concatenate([ac.array for ac in asr_input])
    wav_length = 393920
    opus_is_alittle_longer = 70
    assert audio_signal.shape[0] == wav_length + opus_is_alittle_longer
    print(
        f"got audio of {sum(len(a.array) for a in asr_input)/expected_sample_rate:.2f} seconds"
    )

    start_time = time()
    streaming_asr = Aschinglupi(
        hf_asr_decoding_inferencer=asr_decode_inferencer,
        transcript_gluer=TranscriptGluer(),
        audio_bufferer=OverlapArrayChunker(
            chunk_size=int(window_dur * SR),
            minimum_chunk_size=int(1 * SR),  # one second!
            min_step_size=int(step_dur * SR),
        ),
    ).build()
    startup_time = time() - start_time
    start_time = time()

    streaming_asr.reset()

    outputs: list[ASRStreamInferenceOutput] = [
        t for inpt in asr_input for t in streaming_asr.handle_inference_input(inpt)
    ]
    transcript = ""
    for tr in outputs:
        transcript = remove_and_append(transcript, tr.ending_to_be_removed, tr.text)
    transcript = transcript.strip(" ")
    assert len(outputs) == num_responses
    assert outputs[-1].end_of_message
    at: AlignedTranscript = outputs[-1].aligned_transcript
    inference_duration = time() - start_time
    hyp = at.text.strip(" ")
    assert hyp == transcript

    ref = normalize_filter_text(
        librispeech_raw_ref,
        asr_decode_inferencer.logits_inferencer.vocab,
        text_cleaner="en",
        casing=Casing.upper,
    )
    # print(f"{ref=}")
    # print(f"{hyp=}")
    diff_line = smithwaterman_aligned_icdiff(ref, hyp)
    print(f"{window_dur=},{step_dur=}")

    print(diff_line)
    cer = calc_cer([(hyp, ref)])
    print(
        f"CER: {cer},start-up took: {startup_time}, inference took: {inference_duration} seconds"
    )
    assert cer <= max_CER

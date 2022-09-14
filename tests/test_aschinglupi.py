import numpy as np
import pytest
from time import time

from conftest import get_test_cache_base
from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.asr_inference.asr_chunk_infer_glue_pipeline import Aschinglupi, \
    gather_final_aligned_transcripts
from ml4audio.asr_inference.hfwav2vec2_asr_decode_inferencer import \
    HFASRDecodeInferencer
from ml4audio.asr_inference.transcript_gluer import TranscriptGluer, \
    ASRStreamInferenceOutput
from ml4audio.audio_utils.overlap_array_chunker import (
    audio_messages_from_file,
    OverlapArrayChunker,
)
from ml4audio.text_processing.asr_text_normalization import (
    normalize_filter_text,
    Casing,
)
from ml4audio.text_processing.metrics_calculation import calc_cer
from ml4audio.text_processing.pretty_diff import smithwaterman_aligned_icdiff

BASE_PATHES["asr_inference"] = get_test_cache_base()


@pytest.mark.parametrize(
    "step_dur,window_dur,max_CER",
    [
        (1.0, 2.0, 0.062),
        (1.0, 4.0, 0.017),
        (4.0, 8.0, 0.01),
        (1.0, 8.0, 0.023),
    ],
)
def test_ASRStreamInferencer(
    asr_decode_inferencer: HFASRDecodeInferencer,
    librispeech_audio_file,
    librispeech_raw_ref,
    step_dur: float,
    window_dur: float,
    max_CER: float,
):

    SR = expected_sample_rate = asr_decode_inferencer.logits_inferencer.input_sample_rate
    asr_input = list(
        audio_messages_from_file(librispeech_audio_file, expected_sample_rate)
    )
    assert asr_input[-1].end_of_signal
    audio_signal = np.concatenate([ac.array for ac in asr_input])
    assert audio_signal.shape[0] == 393920

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
    outp: ASRStreamInferenceOutput = list(
        gather_final_aligned_transcripts(streaming_asr, asr_input)
    )[0]
    inference_duration = time() - start_time
    hyp = outp.aligned_transcript.text.strip(" ")

    ref = normalize_filter_text(
        librispeech_raw_ref,
        asr_decode_inferencer.logits_inferencer.vocab,
        text_normalizer="en",
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
    assert cer < max_CER

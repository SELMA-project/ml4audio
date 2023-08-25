from time import time

import icdiff
import pytest

from ctc_asr_chunked_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from ml4audio.audio_utils.audio_io import load_and_resample_16bit_PCM
from ml4audio.text_processing.asr_metrics import calc_cer


@pytest.mark.parametrize(
    "asr_hf_inferencer,max_CER",
    [
        ((16_000, "greedy"), 0.0),  # WTF! this model reaches 0% CER! overfitted?
        ((8_000, "greedy"), 0.0033),
        ((4_000, "greedy"), 0.091),
        ((4_000, "beamsearch"), 0.02),
        ((8_000, "beamsearch"), 0.007),
    ],
    indirect=["asr_hf_inferencer"],
)
def test_HFASRDecodeInferencer(
    asr_hf_inferencer: HFASRDecodeInferencer,
    librispeech_audio_file,
    librispeech_raw_ref,
    max_CER,
):

    expected_sample_rate = asr_hf_inferencer.input_sample_rate
    audio_array = load_and_resample_16bit_PCM(
        librispeech_audio_file, expected_sample_rate
    )

    start_time = time()
    transcript = asr_hf_inferencer.transcribe_audio_array(audio_array.squeeze())
    inference_duration = time() - start_time
    hyp = transcript.letters
    cd = icdiff.ConsoleDiff(cols=120)
    diff_line = "\n".join(
        cd.make_table(
            [librispeech_raw_ref],
            [hyp],
            "ref",
            "hyp",
        )
    )
    print(diff_line)
    cer = calc_cer([(hyp, librispeech_raw_ref)])
    print(f"CER: {cer}, inference took: {inference_duration} seconds")
    assert cer <= max_CER

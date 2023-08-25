from dataclasses import dataclass
from time import time

import icdiff
import pytest

from ctc_asr_chunked_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from ml4audio.audio_utils.audio_io import load_and_resample_16bit_PCM
from ml4audio.text_processing.asr_metrics import calc_cer
from tests.conftest import TestParams


@pytest.mark.parametrize(
    "asr_hf_inferencer,max_CER",
    [
        (TestParams(), 0.0),  # WTF! this model reaches 0% CER! overfitted?
        (TestParams(8_000), 0.0033),
        (TestParams(4_000), 0.091),
        (TestParams(4_000, decoder_name="beamsearch"), 0.02),
        (TestParams(8_000, decoder_name="beamsearch"), 0.007),
        (
            TestParams(inferencer_name="nemo-conformer", decoder_name="beamsearch"),
            0.094,
        ),
        # (TestParams(inferencer_name="nemo-conformer",decoder_name="beamsearch-stupidlm"), 0.094),
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
    hyp = transcript.letters.upper()
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
    decoder_name = asr_hf_inferencer.decoder.__class__.__name__
    print(
        f"{asr_hf_inferencer.logits_inferencer.name},{decoder_name}\tCER: {cer}, inference took: {inference_duration} seconds"
    )
    assert cer <= max_CER

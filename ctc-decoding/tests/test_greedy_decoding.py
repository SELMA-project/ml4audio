import icdiff
import numpy as np

from ctc_decoding.huggingface_ctc_decoding import (
    HFCTCGreedyDecoder,
)
from ml4audio.text_processing.asr_metrics import calc_cer

TARGET_SAMPLE_RATE = 16000


def test_GreedyDecoder(
    hfwav2vec2_base_tokenizer,
    librispeech_logtis_file,
    librispeech_ref,
):
    logits = np.load(librispeech_logtis_file, allow_pickle=True)
    decoder = HFCTCGreedyDecoder(
        tokenizer_name_or_path="facebook/wav2vec2-base-960h",
    ).build()
    transcript = decoder.ctc_decode(logits.squeeze())[0]
    hyp = transcript.text


    cer = calc_cer([librispeech_ref],[hyp])
    assert cer == 0.0

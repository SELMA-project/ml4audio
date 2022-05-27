import icdiff
import numpy as np
import torch

from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from ml4audio.text_processing.ctc_decoding import GreedyDecoder
from ml4audio.text_processing.metrics_calculation import calc_cer

TARGET_SAMPLE_RATE = 16000


def test_GreedyDecoder(
    vocab,
    librispeech_logtis_file,
    librispeech_ref,
):

    logits = np.load(librispeech_logtis_file, allow_pickle=True)
    print(f"{logits.shape=}")
    tn = TranscriptNormalizer(casing=Casing.upper, text_normalizer="en", vocab=vocab)
    decoder = GreedyDecoder(vocab=tn.vocab).build()
    transcript = decoder.decode_batch(torch.from_numpy(logits))[0][0]

    hyp = transcript.text
    cd = icdiff.ConsoleDiff(cols=120)
    diff_line = "\n".join(
        cd.make_table(
            [librispeech_ref],
            [hyp],
            "ref",
            "hyp",
        )
    )
    print(diff_line)
    cer = calc_cer([(hyp, librispeech_ref)])
    assert cer == 0.0

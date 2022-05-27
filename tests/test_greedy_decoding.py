import icdiff
import numpy as np
import torch
from transformers import Wav2Vec2CTCTokenizer

from ml4audio.text_processing.ctc_decoding import GreedyDecoder
from ml4audio.text_processing.metrics_calculation import calc_cer

TARGET_SAMPLE_RATE = 16000


def test_GreedyDecoder(
    vocab,
    librispeech_logtis_file,
    librispeech_ref,
):
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    logits = np.load(librispeech_logtis_file, allow_pickle=True)
    decoder = GreedyDecoder(tokenizer=tokenizer)
    transcript = decoder.decode(torch.from_numpy(logits.squeeze()))[0]
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

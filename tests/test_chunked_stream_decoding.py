import os
import shutil

import numpy as np
import pytest

from conftest import (
    get_test_vocab,
    TEST_RESOURCES,
    overlapping_messages_from_array,
    assert_transcript_cer,
    load_hfwav2vec2_base_tokenizer,
)
from misc_utils.prefix_suffix import PrefixSuffix
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from ml4audio.text_processing.lm_model_for_pyctcdecode import (
    KenLMForPyCTCDecodeFromArpa,
)
from ml4audio.text_processing.streaming_beam_search_decoder import ChunkedPyctcDecoder

TARGET_SAMPLE_RATE = 16000

# TODO: this is very ugly
cache_base = PrefixSuffix("pwd", "/tmp/cache")
shutil.rmtree(str(cache_base), ignore_errors=True)
os.makedirs(str(cache_base))

tn = TranscriptNormalizer(
    casing=Casing.upper, text_normalizer="en", vocab=get_test_vocab()
)


@pytest.mark.skip(reason="not implemented yet! -> you want it you fix it!")
def test_chunked_streaming_beam_search_decoder(
    librispeech_logtis_file,
    librispeech_ref,
):

    logits = np.load(librispeech_logtis_file, allow_pickle=True).squeeze()
    logits_chunks: list[MessageChunk] = list(
        overlapping_messages_from_array(logits, step_size=100, chunk_size=200)
    )

    max_cer = 0.007
    decoder: ChunkedPyctcDecoder = ChunkedPyctcDecoder(
        lm_weight=1.0,
        beta=0.5,
        beam_size=100,
        tokenizer_name_or_path="facebook/wav2vec2-base-960h",
        lm_data=KenLMForPyCTCDecodeFromArpa(
            name="test",
            cache_base=cache_base,
            arpa_file=f"{TEST_RESOURCES}/lm.arpa",
            transcript_normalizer=tn,
        ),
    ).build()

    for ch in logits_chunks:
        nonfinal = decoder.decode(ch)
        # print(f"{final[0].text=}")
        print(f"{nonfinal[0].text=}")
    ref = librispeech_ref
    hyp = nonfinal[0].text
    assert_transcript_cer(hyp, ref, max_cer)

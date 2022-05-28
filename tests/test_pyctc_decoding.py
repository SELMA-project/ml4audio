import os
import shutil

import numpy as np
import pytest
import torch

from ml4audio.text_processing.lm_model_for_pyctcdecode import (
    KenLMForPyCTCDecodeFromArpa,
    KenLMForPyCTCDecodeFromArpaCorpus,
    KenLMForPyCTCDecode,
)
from ml4audio.text_processing.pyctc_decoder import PyCTCKenLMDecoder
from misc_utils.buildable import BuildableList
from misc_utils.prefix_suffix import PrefixSuffix
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from ml4audio.text_processing.kenlm_arpa import ArpaBuilder, ArpaArgs
from ml4audio.text_processing.word_based_text_corpus import (
    WordBasedLMCorpus,
    RglobRawCorpus,
)
from conftest import get_test_vocab, TEST_RESOURCES, load_hfwav2vec2_base_tokenizer
from conftest import assert_transcript_cer

TARGET_SAMPLE_RATE = 16000

# TODO: this is very ugly
cache_base = PrefixSuffix("pwd", "/tmp/cache")
shutil.rmtree(str(cache_base), ignore_errors=True)
os.makedirs(str(cache_base))

tn = TranscriptNormalizer(
    casing=Casing.upper, text_normalizer="en", vocab=get_test_vocab()
)


@pytest.mark.parametrize(
    "lm_data,max_cer",
    [
        (
            KenLMForPyCTCDecodeFromArpa(
                name="test",
                cache_base=cache_base,
                arpa_file=f"{TEST_RESOURCES}/lm.arpa",
                transcript_normalizer=tn,
            ),
            0.007,
        ),
        (
            KenLMForPyCTCDecodeFromArpaCorpus(
                cache_base=cache_base,
                transcript_normalizer=tn,
                arpa_builder=ArpaBuilder(
                    cache_base=cache_base,
                    arpa_args=ArpaArgs(
                        order=5,
                        prune="|".join(str(k) for k in [0, 8, 16]),
                    ),
                    corpus=WordBasedLMCorpus(
                        name="test",
                        cache_base=cache_base,
                        raw_corpora=BuildableList[RglobRawCorpus](
                            [
                                RglobRawCorpus(
                                    cache_base=cache_base,
                                    corpus_dir=TEST_RESOURCES,
                                    file_pattern="*corpus.txt",
                                )
                            ]
                        ),
                        normalizer=tn,
                    ),
                ),
            ),
            0.0035,
        ),
    ],
)
def test_PyCTCKenLMDecoder(
    hfwav2vec2_base_tokenizer,
    lm_data: KenLMForPyCTCDecode,
    max_cer: float,
    librispeech_logtis_file,
    librispeech_ref,
):

    logits = np.load(librispeech_logtis_file, allow_pickle=True)

    decoder = PyCTCKenLMDecoder(
        tokenizer=hfwav2vec2_base_tokenizer,
        lm_weight=1.0,
        beta=0.5,
        lm_data=lm_data,
    )
    decoder.build()
    transcript = decoder.decode(torch.from_numpy(logits.squeeze()))[0]
    ref = librispeech_ref
    hyp = transcript.text
    assert_transcript_cer(hyp, ref, max_cer)


@pytest.mark.parametrize(
    "decoder",
    [
        (
            PyCTCKenLMDecoder(
                tokenizer=load_hfwav2vec2_base_tokenizer(),
                lm_weight=1.0,
                beta=0.5,
                lm_data=KenLMForPyCTCDecodeFromArpa(
                    name="test",
                    cache_base=cache_base,
                    arpa_file=f"{TEST_RESOURCES}/lm.arpa",
                    transcript_normalizer=tn,
                ),
            )
        ),
    ],
)
def test_decoders(
    decoder,
    librispeech_logtis_file,
    librispeech_ref,
):
    max_cer = 0.007

    logits = np.load(librispeech_logtis_file, allow_pickle=True)
    decoder.build()
    transcript = decoder.decode(torch.from_numpy(logits.squeeze()))[0]
    ref = librispeech_ref
    hyp = transcript.text
    assert_transcript_cer(hyp, ref, max_cer)

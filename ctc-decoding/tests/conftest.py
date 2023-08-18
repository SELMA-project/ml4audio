import os
import sys

import icdiff

from ml4audio.audio_utils.test_utils import get_test_vocab, TEST_RESOURCES
from ml4audio.text_processing.asr_metrics import calc_cer

sys.path.append(os.path.dirname(__file__))  # TODO: WTF! this is a hack!

from data_io.readwrite_files import read_lines
from misc_utils.beartypes import NeStr
from ml4audio.text_processing.asr_text_normalization import (
    normalize_filter_text,
    Casing,
)

from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from transformers import Wav2Vec2CTCTokenizer

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

import pytest


@pytest.fixture
def vocab():
    return get_test_vocab()


@pytest.fixture
def hfwav2vec2_base_tokenizer():
    return load_hfwav2vec2_base_tokenizer()


def load_hfwav2vec2_base_tokenizer():
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer


@pytest.fixture
def librispeech_logtis_file():
    return (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_logits.npy"
    )


@pytest.fixture
def librispeech_ref(vocab) -> NeStr:
    ref_txt = (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    )
    raw_ref = next(iter(read_lines(ref_txt)))
    ref = normalize_filter_text(raw_ref, vocab, text_cleaner="en", casing=Casing.upper)
    return ref


def assert_transcript_cer(hyp, ref, max_cer):
    cd = icdiff.ConsoleDiff(cols=120)
    diff_line = "\n".join(
        cd.make_table(
            [ref],
            [hyp],
            "ref",
            "hyp",
        )
    )
    print(diff_line)
    cer = calc_cer([(hyp, ref)])
    print(f"{cer=}")
    assert cer < max_cer

import os
import sys

from ml4audio.audio_utils.test_utils import TEST_RESOURCES

sys.path.append(os.path.dirname(__file__))  # TODO: WTF! this is a hack!

from data_io.readwrite_files import read_lines

from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from transformers import Wav2Vec2CTCTokenizer

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

import pytest


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
def librispeech_ref():
    ref_txt = (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    )
    raw_ref = next(iter(read_lines(ref_txt)))
    return raw_ref

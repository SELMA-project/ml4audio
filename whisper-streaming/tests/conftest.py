from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.audio_utils.test_utils import (
    get_test_cache_base,
    TEST_RESOURCES,
)

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

from data_io.readwrite_files import read_lines
import pytest

cache_base = get_test_cache_base()
BASE_PATHES["cache_root"] = cache_base


@pytest.fixture
def librispeech_ref():
    ref_txt = (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    )
    raw_ref = next(iter(read_lines(ref_txt)))
    return raw_ref


@pytest.fixture
def librispeech_audio_file():
    return f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"

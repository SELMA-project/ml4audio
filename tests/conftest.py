# sys.path.append(os.path.dirname(__file__))  # TODO: WTF! this is a hack!
from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

from ml4audio.audio_utils.test_utils import get_test_vocab, TEST_RESOURCES

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

import pytest


@pytest.fixture
def vocab():
    return get_test_vocab()


@pytest.fixture
def arpa_file():
    return f"{TEST_RESOURCES}/lm.arpa"

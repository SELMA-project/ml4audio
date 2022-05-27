# sys.path.append(os.path.dirname(__file__))  # TODO: WTF! this is a hack!
from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

from misc_utils.beartypes import NumpyInt16Dim1, NeStr
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import HfCheckpoint
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)
from ml4audio.audio_utils.audio_io import read_audio_chunks_from_file

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

from data_io.readwrite_files import read_lines, read_jsonl
import pytest
from ml4audio.text_processing.asr_text_normalization import (
    normalize_filter_text,
    Casing,
)
import os
import shutil

from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix

TEST_RESOURCES = "tests/resources"
BASE_PATHES["test_resources"] = TEST_RESOURCES


def get_test_cache_base():
    cache_base = PrefixSuffix("test_resources", "cache")
    if os.path.isdir(str(cache_base)):
        shutil.rmtree(str(cache_base))
    os.makedirs(str(cache_base))
    return cache_base


def get_test_vocab():
    return f"""<pad>
<s>
</s>
<unk>
|
E
T
A
O
N
I
H
S
R
D
L
U
M
W
C
F
G
Y
P
B
V
K
'
X
J
Q
Z""".split(
        "\n"
    )


@pytest.fixture
def vocab():
    return get_test_vocab()


@pytest.fixture
def hfwav2vec2_base_logits_inferencer(request):
    cache_base = get_test_cache_base()

    if not hasattr(request, "param"):
        expected_sample_rate = 16000
    else:
        expected_sample_rate = request.param

    model = "facebook/wav2vec2-base-960h"
    logits_inferencer = HFWav2Vec2LogitsInferencer(
        checkpoint=HfCheckpoint(
            name=model, model_name_or_path=model, cache_base=cache_base
        ),
        input_sample_rate=expected_sample_rate,
    ).build()
    return logits_inferencer




# TODO!!
@pytest.fixture
def librispeech_ref(vocab) -> NeStr:
    ref_txt = (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    )
    raw_ref = next(iter(read_lines(ref_txt)))
    ref = normalize_filter_text(
        raw_ref, vocab, text_normalizer="en", casing=Casing.upper
    )
    return ref


@pytest.fixture
def librispeech_raw_ref():
    ref_txt = (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    )
    raw_ref = next(iter(read_lines(ref_txt)))
    return raw_ref


@pytest.fixture
def german_tuda_raw_ref():
    raw_ref = next(read_lines(f"{TEST_RESOURCES}/tuda_2015-02-03-13-51-36_ref.txt"))
    return raw_ref


@pytest.fixture
def librispeech_logtis_file():
    return (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_logits.npy"
    )


@pytest.fixture
def arpa_file():
    return f"{TEST_RESOURCES}/lm.arpa"


@pytest.fixture
def librispeech_audio_file():
    return f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011.wav"


@pytest.fixture
def expected_starts_ends():
    start_ends = [
        (d["start"], d["end"])
        for d in read_jsonl(f"{TEST_RESOURCES}/aligned_segments.jsonl")
    ]
    return [list(x) for x in zip(*start_ends)]


@pytest.fixture
def librispeech_audio_chunks(librispeech_audio_file) -> list[NumpyInt16Dim1]:
    audio_chunks = list(
        read_audio_chunks_from_file(
            librispeech_audio_file, target_sample_rate=16_000, chunk_duration=0.1
        )
    )
    return audio_chunks

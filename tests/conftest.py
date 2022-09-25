# sys.path.append(os.path.dirname(__file__))  # TODO: WTF! this is a hack!
from warnings import filterwarnings

import icdiff
from beartype import beartype
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from numpy.typing import NDArray
from transformers import Wav2Vec2CTCTokenizer

from misc_utils.beartypes import NumpyInt16Dim1, NeStr, NumpyFloat1DArray
from ml4audio.asr_inference.hfwav2vec2_asr_decode_inferencer import \
    HFASRDecodeInferencer
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import HfCheckpoint, \
    VocabFromASRLogitsInferencer
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)
from ml4audio.audio_utils.audio_io import (
    read_audio_chunks_from_file,
    convert_to_16bit_array,
    break_array_into_chunks,
)
from ml4audio.audio_utils.overlap_array_chunker import (
    OverlapArrayChunker,
    audio_messages_from_chunks,
    messages_from_chunks,
)
from ml4audio.text_processing.ctc_decoding import GreedyDecoder
from ml4audio.text_processing.asr_metrics import calc_cer

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
def hfwav2vec2_base_tokenizer():
    return load_hfwav2vec2_base_tokenizer()


def load_hfwav2vec2_base_tokenizer():
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer


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

@pytest.fixture
def asr_decode_inferencer(request):
    cache_base = get_test_cache_base()

    if not hasattr(request, "param"):
        expected_sample_rate = 16000
    else:
        expected_sample_rate = request.param

    model = "facebook/wav2vec2-base-960h"
    # model = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    logits_inferencer = HFWav2Vec2LogitsInferencer(
        checkpoint=HfCheckpoint(
            name=model, model_name_or_path=model, cache_base=cache_base
        ),
        input_sample_rate=expected_sample_rate,
    )
    asr = HFASRDecodeInferencer(
        logits_inferencer=logits_inferencer,
        decoder=GreedyDecoder(tokenizer_name_or_path=model),
    )
    asr.build()
    return asr


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


@beartype
def overlapping_audio_messages_from_audio_array(
    audio_array: NumpyFloat1DArray, sr: int, step_dur: float, chunk_dur: float
):
    chunker = OverlapArrayChunker(
        chunk_size=int(chunk_dur * sr),
        min_step_size=int(step_dur * sr),
    )
    chunker.reset()

    audio_array = convert_to_16bit_array(audio_array)
    small_chunks = break_array_into_chunks(audio_array, int(sr * 0.1))
    chunks_g = (
        am
        for ch in audio_messages_from_chunks("dummy-id", small_chunks)
        for am in chunker.handle_datum(ch)
    )
    yield from chunks_g


@beartype
def overlapping_messages_from_array(
    audio_array: NDArray, step_size: int, chunk_size: int
):
    chunker = OverlapArrayChunker(
        chunk_size=chunk_size,
        min_step_size=step_size,
    )
    chunker.reset()

    small_chunks = break_array_into_chunks(audio_array, chunk_size=100)
    chunks_g = (
        am
        for ch in messages_from_chunks("dummy-id", small_chunks)
        for am in chunker.handle_datum(ch)
    )
    yield from chunks_g

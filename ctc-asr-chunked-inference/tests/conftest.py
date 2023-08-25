import os
import sys

from ml4audio.asr_inference.logits_inferencer.huggingface_checkpoints import (
    HfModelFromCheckpoint,
)
from ml4audio.audio_utils.test_utils import (
    get_test_vocab,
    get_test_cache_base,
    TEST_RESOURCES,
)

sys.path.append(os.path.dirname(__file__))  # TODO: WTF! this is a hack!
from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

from ctc_asr_chunked_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from ctc_decoding.huggingface_ctc_decoding import HFCTCGreedyDecoder
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

from data_io.readwrite_files import read_lines
import pytest


@pytest.fixture
def vocab():
    return get_test_vocab()


@pytest.fixture
def asr_decode_inferencer(request):
    cache_base = get_test_cache_base()

    if not hasattr(request, "param"):
        input_sample_rate = 16000
    else:
        input_sample_rate = request.param

    model = "facebook/wav2vec2-base-960h"
    logits_inferencer = HFWav2Vec2LogitsInferencer(
        checkpoint=HfModelFromCheckpoint(
            name=model,
            model_name_or_path=model,
            hf_model_type="Wav2Vec2ForCTC",
            base_dir=cache_base,
        ),
    )
    asr = HFASRDecodeInferencer(
        logits_inferencer=logits_inferencer,
        decoder=HFCTCGreedyDecoder(tokenizer_name_or_path=model),
        input_sample_rate=input_sample_rate,
    )
    asr.build()
    return asr


@pytest.fixture
def librispeech_raw_ref():
    ref_txt = (
        f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    )
    raw_ref = next(iter(read_lines(ref_txt)))
    return raw_ref


@pytest.fixture
def librispeech_audio_file():
    return f"{TEST_RESOURCES}/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"

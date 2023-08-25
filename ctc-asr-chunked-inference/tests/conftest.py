import os
import sys
from dataclasses import dataclass

from ctc_decoding.lm_model_for_pyctcdecode import GzippedArpaAndUnigramsForPyCTCDecode
from ctc_decoding.pyctc_decoder import PyCTCKenLMDecoder
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    ASRLogitsInferencer,
)
from ml4audio.asr_inference.logits_inferencer.huggingface_checkpoints import (
    HfModelFromCheckpoint,
)
from ml4audio.asr_inference.logits_inferencer.nemo_asr_logits_inferencer import (
    NemoASRLogitsInferencer,
)
from ml4audio.audio_utils.test_utils import (
    get_test_vocab,
    get_test_cache_base,
    TEST_RESOURCES,
)
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from ml4audio.text_processing.kenlm_arpa import AnArpaFile

sys.path.append(os.path.dirname(__file__))  # TODO: WTF! this is a hack!
from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

from ctc_asr_chunked_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from ctc_decoding.huggingface_ctc_decoding import (
    HFCTCGreedyDecoder,
    VocabFromHFTokenizer,
)
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

from data_io.readwrite_files import read_lines
import pytest

cache_base = get_test_cache_base()

TEST_MODEL_NAME = "facebook/wav2vec2-base-960h"


@dataclass(frozen=True)
class TestParams:
    input_sample_rate: int = 16000
    inferencer_name: str = "hf-wav2vec2"
    decoder_name: str = "greedy"


@pytest.fixture
def vocab():
    return get_test_vocab()


def build_decoder(name: str, vocab: list[str]):
    NAME2DECODER = {
        "greedy": HFCTCGreedyDecoder(tokenizer_name_or_path=TEST_MODEL_NAME),
        "beamsearch": PyCTCKenLMDecoder(
            vocab=vocab,
            lm_weight=1.0,
            beta=0.5,
            ngram_lm_model=GzippedArpaAndUnigramsForPyCTCDecode(
                base_dir=cache_base,
                raw_arpa=AnArpaFile(arpa_filepath=f"{TEST_RESOURCES}/lm.arpa"),
                transcript_normalizer=TranscriptNormalizer(
                    casing=Casing.upper, text_cleaner="en", vocab=vocab
                ),
            ),
        ),
        "beamsearch-stupidlm": PyCTCKenLMDecoder(
            vocab=vocab,
            lm_weight=1.0,
            beta=0.5,
            ngram_lm_model=GzippedArpaAndUnigramsForPyCTCDecode(
                base_dir=cache_base,
                raw_arpa=AnArpaFile(arpa_filepath=f"{TEST_RESOURCES}/stupid_lm.arpa"),
                transcript_normalizer=TranscriptNormalizer(
                    casing=Casing.upper, text_cleaner="en", vocab=vocab
                ),
            ),
        ),
    }
    return NAME2DECODER[name]


def build_logits_inferencer(name: str) -> ASRLogitsInferencer:
    # SMALL_CTC_CONFORMER = "nvidia/stt_en_conformer_ctc_small"
    SMALL_CTC_CONFORMER = "stt_en_conformer_ctc_small"
    NAME2INFERENCER = {
        "hf-wav2vec2": HFWav2Vec2LogitsInferencer(
            checkpoint=HfModelFromCheckpoint(
                name=TEST_MODEL_NAME,
                model_name_or_path=TEST_MODEL_NAME,
                hf_model_type="Wav2Vec2ForCTC",
                base_dir=cache_base,
            ),
        ),
        "nemo-conformer": NemoASRLogitsInferencer(SMALL_CTC_CONFORMER),
    }
    return NAME2INFERENCER[name].build()


@pytest.fixture
def asr_hf_inferencer(request):

    tp: TestParams = request.param

    inferencer = build_logits_inferencer(tp.inferencer_name)
    asr = HFASRDecodeInferencer(
        logits_inferencer=inferencer,
        decoder=build_decoder(tp.decoder_name, inferencer.vocab),
        input_sample_rate=tp.input_sample_rate,
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

import os
import shutil

import kenlm
import numpy as np
import pytest

from conftest import (
    get_test_vocab,
    TEST_RESOURCES,
    load_hfwav2vec2_base_tokenizer,
    overlapping_messages_from_array,
    assert_transcript_cer,
)
from data_io.readwrite_files import read_lines
from misc_utils.prefix_suffix import PrefixSuffix
from ml4audio.asr_inference.logits_cutter import LogitsCutter
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from ml4audio.text_processing.lm_model_for_pyctcdecode import (
    KenLMForPyCTCDecodeFromArpa,
)
from ml4audio.text_processing.pyctc_decoder import OutputBeamDc
from ml4audio.text_processing.streaming_beam_search_decoder import (
    StreamingBeamSearchDecoderCTC,
    IncrBeam,
)
from pyctcdecode import Alphabet, LanguageModel
from pyctcdecode.constants import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_HOTWORD_WEIGHT,
)
from pyctcdecode.language_model import HotwordScorer

TARGET_SAMPLE_RATE = 16000

# TODO: this is very ugly
cache_base = PrefixSuffix("pwd", "/tmp/cache")
shutil.rmtree(str(cache_base), ignore_errors=True)
os.makedirs(str(cache_base))

tn = TranscriptNormalizer(
    casing=Casing.upper, text_normalizer="en", vocab=get_test_vocab()
)


lm_data: KenLMForPyCTCDecodeFromArpa = KenLMForPyCTCDecodeFromArpa(
    name="test",
    cache_base=cache_base,
    arpa_file=f"{TEST_RESOURCES}/lm.arpa",
    transcript_normalizer=tn,
).build()
unigrams = list(read_lines(lm_data.unigrams_filepath))


@pytest.mark.parametrize(
    "decoder",
    [
        (
            StreamingBeamSearchDecoderCTC(
                Alphabet.build_alphabet(
                    list(load_hfwav2vec2_base_tokenizer().get_vocab().keys())
                ),
                language_model=LanguageModel(
                    kenlm_model=kenlm.Model(lm_data.arpa_filepath),
                    unigrams=unigrams,
                    alpha=1.0,
                    beta=0.5,
                    # unk_score_offset=unk_score_offset,
                    # score_boundary=lm_score_boundary,
                ),
            )
        ),
    ],
)
def test_chunked_streaming_beam_search_decoder(
    decoder,
    librispeech_logtis_file,
    librispeech_ref,
):

    logits = np.load(librispeech_logtis_file, allow_pickle=True).squeeze()
    logits_chunks: list[MessageChunk] = list(
        overlapping_messages_from_array(logits, step_size=100, chunk_size=200)
    )
    chunk_spans = [
        (ch.array, (ch.frame_idx, ch.frame_idx + len(ch.array))) for ch in logits_chunks
    ]

    lc = LogitsCutter()
    lc.reset()

    left_right = [lc.calc_left_right(l, s_e) for l, s_e in chunk_spans]
    print(f"{[(l.shape if l is not None else 0,r.shape) for l,r in left_right ]=}")
    parts = [l for l, r in left_right] + [left_right[-1][1]]
    parts = [x for x in parts if x is not None]

    max_cer = 0.007

    beams_g = decoder._decode_logits(
        logits=None,
        beam_width=DEFAULT_BEAM_WIDTH,
        beam_prune_logp=DEFAULT_PRUNE_LOGP,
        token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
        prune_history=DEFAULT_PRUNE_BEAMS,
        hotword_scorer=HotwordScorer.build_scorer(
            hotwords=None, weight=DEFAULT_HOTWORD_WEIGHT
        ),
        lm_start_state=None,
    )
    print(f"{beams_g.send(None)=}")

    frame_idx = 0
    for logit_chunk in parts:
        for logits_col in logit_chunk:
            incr_beams: list[IncrBeam] = beams_g.send((frame_idx, logits_col))
            frame_idx += 1
        print(f"{frame_idx=}, {incr_beams[0].text=}")

    ref = librispeech_ref
    hyp = incr_beams[0].text
    assert_transcript_cer(hyp, ref, max_cer)

    incr_beams = next(beams_g)
    beams = [OutputBeamDc(*b) for b in incr_beams]

    ref = librispeech_ref
    hyp = beams[0].text
    assert_transcript_cer(hyp, ref, max_cer)

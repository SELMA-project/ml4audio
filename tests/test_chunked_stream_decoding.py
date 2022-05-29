import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Union, Optional

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
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.prefix_suffix import PrefixSuffix
from ml4audio.asr_inference.logits_cutter import LogitsCutter
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from ml4audio.text_processing.lm_model_for_pyctcdecode import (
    KenLMForPyCTCDecodeFromArpa,
    KenLMForPyCTCDecode,
)
from ml4audio.text_processing.streaming_beam_search_decoder import (
    StreamingBeamSearchDecoderCTC,
    IncrBeam,
    DecodeParams,
)
from pyctcdecode import Alphabet, LanguageModel
from pyctcdecode.constants import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_HOTWORD_WEIGHT,
)
from pyctcdecode.decoder import EMPTY_START_BEAM, Beam
from pyctcdecode.language_model import HotwordScorer

TARGET_SAMPLE_RATE = 16000

# TODO: this is very ugly
cache_base = PrefixSuffix("pwd", "/tmp/cache")
shutil.rmtree(str(cache_base), ignore_errors=True)
os.makedirs(str(cache_base))

tn = TranscriptNormalizer(
    casing=Casing.upper, text_normalizer="en", vocab=get_test_vocab()
)


@dataclass
class ChunkedPyctcDecoder(Buildable):

    lm_weight: Union[_UNDEFINED, float] = UNDEFINED
    beta: Union[_UNDEFINED, float] = UNDEFINED
    # cannot do this with beartype NeList[str] for vocab, cause it might be a CachedList
    # vocab: Union[_UNDEFINED, list[str]] = UNDEFINED

    lm_data: Union[
        KenLMForPyCTCDecode, _UNDEFINED
    ] = UNDEFINED  # TODO: rename lm_data to lm_model

    num_best: int = 1  # number of beams to return
    beam_size: int = 100

    _pyctc_decoder: Optional[StreamingBeamSearchDecoderCTC] = field(
        init=False, repr=False, default=None
    )

    def _build_self(self) -> Any:
        self.lc = LogitsCutter()
        self.lc.reset()
        unigrams = list(read_lines(self.lm_data.unigrams_filepath))

        self._pyctc_decoder = StreamingBeamSearchDecoderCTC(
            Alphabet.build_alphabet(
                list(load_hfwav2vec2_base_tokenizer().get_vocab().keys())
            ),
            language_model=LanguageModel(
                kenlm_model=kenlm.Model(self.lm_data.arpa_filepath),
                unigrams=unigrams,
                alpha=self.lm_weight,
                beta=self.beta,
                # unk_score_offset=unk_score_offset,
                # score_boundary=lm_score_boundary,
            ),
        )

        language_model = self._pyctc_decoder.model_container[
            self._pyctc_decoder._model_key
        ]
        # start with single beam to expand on
        self.beams = [EMPTY_START_BEAM]
        # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
        self.params = DecodeParams(
            beam_width=DEFAULT_BEAM_WIDTH,
            beam_prune_logp=DEFAULT_PRUNE_LOGP,
            token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
            prune_history=DEFAULT_PRUNE_BEAMS,
            hotword_scorer=HotwordScorer.build_scorer(
                hotwords=None, weight=DEFAULT_HOTWORD_WEIGHT
            ),
            lm_start_state=None,
            language_model=language_model,
        )
        return self

    def decode(self, ch: MessageChunk) -> tuple[list[IncrBeam], list[IncrBeam]]:
        s_e = (ch.frame_idx, ch.frame_idx + len(ch.array))
        left_part, right_part = self.lc.calc_left_right(ch.array, s_e)

        frame_idx = ch.frame_idx
        if left_part is not None:
            for logits_col in left_part:
                self.beams = self._pyctc_decoder.incr_decode_step(
                    self.beams, frame_idx, logits_col, self.params
                )
                frame_idx += 1

            assert frame_idx == ch.frame_idx + len(left_part)

        right_beams: list[Beam] = self.beams
        for k, logits_col in enumerate(right_part):
            right_beams = self._pyctc_decoder.incr_decode_step(
                right_beams, frame_idx + k, logits_col, self.params
            )

        if ch.end_of_signal:
            self.beams = right_beams

        incr_beams = [IncrBeam(*beam) for beam in self.beams]
        incr_right_beams = [IncrBeam(*beam) for beam in right_beams]
        return incr_beams, incr_right_beams


def test_chunked_streaming_beam_search_decoder(
    librispeech_logtis_file,
    librispeech_ref,
):

    logits = np.load(librispeech_logtis_file, allow_pickle=True).squeeze()
    logits_chunks: list[MessageChunk] = list(
        overlapping_messages_from_array(logits, step_size=100, chunk_size=200)
    )

    max_cer = 0.007
    decoder: ChunkedPyctcDecoder = ChunkedPyctcDecoder(
        lm_weight=1.0,
        beta=0.5,
        beam_size=100,
        lm_data=KenLMForPyCTCDecodeFromArpa(
            name="test",
            cache_base=cache_base,
            arpa_file=f"{TEST_RESOURCES}/lm.arpa",
            transcript_normalizer=tn,
        ),
    ).build()

    for ch in logits_chunks:
        final_beams, nonfinal_beams = decoder.decode(ch)
        print(f"{final_beams[0].text=}")
        print(f"{nonfinal_beams[0].text=}")
    ref = librispeech_ref
    hyp = final_beams[0].text
    assert_transcript_cer(hyp, ref, max_cer)

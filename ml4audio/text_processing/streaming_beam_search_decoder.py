from __future__ import division

from dataclasses import dataclass, field
from itertools import islice
from typing import List, Dict, Tuple, Optional, Union, Any

import kenlm
import numpy as np  # type: ignore
from beartype import beartype
from numpy.typing import NDArray

from data_io.readwrite_files import read_lines
from misc_utils.beartypes import NumpyFloat2DArray
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from ml4audio.asr_inference.logits_cutter import LogitsCutter
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk
from ml4audio.text_processing.ctc_decoding import (
    LogitAlignedTranscript,
    BaseCTCDecoder,
    HFCTCDecoder,
)
from ml4audio.text_processing.lm_model_for_pyctcdecode import KenLMForPyCTCDecode
from ml4audio.text_processing.pyctc_decoder import OutputBeamDc
from pyctcdecode import BeamSearchDecoderCTC, Alphabet, LanguageModel
from pyctcdecode.alphabet import BPE_TOKEN
from pyctcdecode.constants import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_HOTWORD_WEIGHT,
)
from pyctcdecode.decoder import (
    LMState,
    EMPTY_START_BEAM,
    Beam,
    NULL_FRAMES,
    _merge_beams,
    _sort_and_trim_beams,
    _normalize_whitespace,
    Frames,
    LMBeam,
)
from pyctcdecode.language_model import HotwordScorer, AbstractLanguageModel


@dataclass
class IncrBeam:
    text: str
    next_word: str
    word_part: str
    last_char: Optional[str]
    text_frames: list[Frames]
    part_frames: Frames
    logit_score: float
    lm_score: float

    def __post_init__(self):
        if len(self.text) > 0:
            assert len(self.text.split(" ")) == len(
                self.text_frames
            ), f"{self.text=}, {self.text_frames=}"


@dataclass
class DecodeParams:
    beam_width: int
    beam_prune_logp: float
    token_min_logp: float
    prune_history: bool
    hotword_scorer: HotwordScorer
    language_model: AbstractLanguageModel
    cached_lm_scores: Optional[Dict[str, Tuple[float, float, LMState]]] = None
    lm_start_state: LMState = None
    cached_p_lm_scores: Dict[str, float] = field(default_factory=lambda: dict())
    force_next_break: bool = False

    def __post_init__(self):
        if self.lm_start_state is None and self.language_model is not None:
            self.cached_lm_scores: Dict[str, Tuple[float, float, LMState]] = {
                "": (0.0, 0.0, self.language_model.get_start_state())
            }
        else:
            self.cached_lm_scores = {"": (0.0, 0.0, self.lm_start_state)}


def _prune_history(beams: List[LMBeam], lm_order: int) -> List[LMBeam]:
    """
    same as : pyctcdecode/decoder.py#147 ;  only change: return LMBeam instead of Beam
    (expects lm-score sorted beams)

    Filter out beams that are the same over max_ngram history.

    Since n-gram language models have a finite history when scoring a new token, we can use that
    fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
    higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
    some amount of beam diversity. If more than the top beam is used in the output it should
    potentially be disabled.
    """
    # let's keep at least 1 word of history
    min_n_history = max(1, lm_order - 1)
    seen_hashes = set()
    filtered_beams = []
    # for each beam after this, check if we need to add it
    for (
        text,
        next_word,
        word_part,
        last_char,
        text_frames,
        part_frames,
        logit_score,
        lm_score,
    ) in beams:
        # hash based on history that can still affect lm scoring going forward
        hash_idx = (tuple(text.split()[-min_n_history:]), word_part, last_char)
        if hash_idx not in seen_hashes:
            filtered_beams.append(
                (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    logit_score,
                    lm_score,
                )
            )
            seen_hashes.add(hash_idx)
    return filtered_beams


class StreamingBeamSearchDecoderCTC(BeamSearchDecoderCTC):
    def _decode_logits(
        self,
        logits: np.ndarray,  # type: ignore [type-arg]
        beam_width: int,
        beam_prune_logp: float,
        token_min_logp: float,
        prune_history: bool,
        hotword_scorer: HotwordScorer,
        lm_start_state: LMState = None,
    ):
        """
        TODO: remove this!
        Perform beam search decoding."""
        # local dictionaries to cache scores during decoding
        # we can pass in an input start state to keep the decoder stateful and working on realtime
        language_model = BeamSearchDecoderCTC.model_container[self._model_key]
        # start with single beam to expand on
        beams = [EMPTY_START_BEAM]
        # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
        params = DecodeParams(
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            prune_history=prune_history,
            hotword_scorer=hotword_scorer,
            lm_start_state=lm_start_state,
            language_model=language_model,
        )

        assert logits is None, f"we don't need this argument!"
        while True:
            inputt = yield [IncrBeam(*beam) for beam in beams]
            if inputt is None:
                break
            frame_idx, logit_col = inputt
            beams = self.incr_decode_step(beams, frame_idx, logit_col, params)

        # final lm scoring and sorting
        output_beams = self.final_scoring_and_sorting(beams, params)
        yield output_beams

    @beartype
    def incr_decode_step(
        self,
        beams: list[LMBeam],
        frame_idx: int,
        logit_col: NDArray,
        params: DecodeParams,
    ) -> list[LMBeam]:
        max_idx = logit_col.argmax()
        idx_list = set(np.where(logit_col >= params.token_min_logp)[0]) | {max_idx}
        new_beams: List[Beam] = []
        for idx_char in idx_list:
            p_char = logit_col[idx_char]
            char = self._idx2vocab[idx_char]
            for (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                logit_score,
                _lm_score,
            ) in beams:
                # if only blank token or same token
                if char == "" or last_char == char:
                    if char == "":
                        new_end_frame = part_frames[0]
                    else:
                        new_end_frame = frame_idx + 1
                    new_part_frames = (
                        part_frames if char == "" else (part_frames[0], new_end_frame)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                        )
                    )
                # if bpe and leading space char
                elif self._is_bpe and (
                    char[:1] == BPE_TOKEN or params.force_next_break
                ):
                    params.force_next_break = False
                    # some tokens are bounded on both sides like ▁⁇▁
                    clean_char = char
                    if char[:1] == BPE_TOKEN:
                        clean_char = clean_char[1:]
                    if char[-1:] == BPE_TOKEN:
                        clean_char = clean_char[:-1]
                        params.force_next_break = True
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            clean_char,
                            char,
                            new_frame_list,
                            (frame_idx, frame_idx + 1),
                            logit_score + p_char,
                        )
                    )
                # if not bpe and space char
                elif not self._is_bpe and char == " ":
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            "",
                            char,
                            new_frame_list,
                            NULL_FRAMES,
                            logit_score + p_char,
                        )
                    )
                # general update of continuing token without space
                else:
                    new_part_frames = (
                        (frame_idx, frame_idx + 1)
                        if part_frames[0] < 0
                        else (part_frames[0], frame_idx + 1)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part + char,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                        )
                    )
        # lm scoring and beam pruning
        new_beams = _merge_beams(new_beams)
        scored_beams = self._get_lm_beams(
            new_beams,
            params.hotword_scorer,
            params.cached_lm_scores,
            params.cached_p_lm_scores,
        )
        # remove beam outliers
        max_score = max([b[-1] for b in scored_beams])
        scored_beams = [
            b for b in scored_beams if b[-1] >= max_score + params.beam_prune_logp
        ]
        # beam pruning by taking highest N prefixes and then filtering down
        trimmed_beams = _sort_and_trim_beams(scored_beams, params.beam_width)
        # prune history and remove lm score from beams
        if params.prune_history:
            lm_order = (
                1 if params.language_model is None else params.language_model.order
            )
            beams = _prune_history(trimmed_beams, lm_order=lm_order)
        else:
            beams = [b for b in trimmed_beams]
        return beams

    @beartype
    def final_scoring_and_sorting(
        self, beams: list[Beam], params: DecodeParams
    ) -> list[OutputBeamDc]:
        """
        hopefully this is not needed! for true streaming-mode, should not be a need to do some final-stuff
        """
        new_beams = []
        for text, _, word_part, _, frame_list, frames, logit_score in beams:
            new_token_times = frame_list if word_part == "" else frame_list + [frames]
            new_beams.append(
                (text, word_part, "", None, new_token_times, (-1, -1), logit_score)
            )
        new_beams = _merge_beams(new_beams)
        scored_beams = self._get_lm_beams(
            new_beams,
            params.hotword_scorer,
            params.cached_lm_scores,
            params.cached_p_lm_scores,
            is_eos=True,
        )
        # remove beam outliers
        max_score = max([b[-1] for b in scored_beams])
        scored_beams = [
            b for b in scored_beams if b[-1] >= max_score + params.beam_prune_logp
        ]
        trimmed_beams = _sort_and_trim_beams(scored_beams, params.beam_width)
        # remove unnecessary information from beams
        output_beams = [
            OutputBeamDc(
                _normalize_whitespace(text),
                params.cached_lm_scores[text][-1]
                if text in params.cached_lm_scores
                else None,
                list(zip(text.split(), text_frames)),
                logit_score,
                lm_score,  # same as logit_score if lm is missing
            )
            for text, _, _, _, text_frames, _, logit_score, lm_score in trimmed_beams
        ]
        return output_beams


@dataclass
class ChunkedPyctcDecoder(HFCTCDecoder):

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
        super()._build_self()

        self.lc = LogitsCutter()
        self.lc.reset()
        unigrams = list(read_lines(self.lm_data.unigrams_filepath))

        self._pyctc_decoder = StreamingBeamSearchDecoderCTC(
            Alphabet.build_alphabet(self.vocab),
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
        EMPTY_START_LMBEAM: LMBeam = ("", "", "", None, [], NULL_FRAMES, 0.0, -1.0)
        self.beams = [EMPTY_START_LMBEAM]
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

    @beartype
    def decode(self, ch: MessageChunk) -> list[LogitAlignedTranscript]:
        """
        # TODO: cut away the "past"
        # currently text can grow infinitely, LM just has limited "context" -> one could limit the keys of params.cached_lm_scores
        """
        s_e = (ch.frame_idx, ch.frame_idx + len(ch.array))
        left_part, right_part = self.lc.calc_left_right(ch.array, s_e)

        frame_idx = ch.frame_idx
        if left_part is not None:
            self.beams = self._decode_array(self.beams, frame_idx, left_part)
            frame_idx = ch.frame_idx + len(left_part)

        right_beams = self._decode_array(self.beams, frame_idx, right_part)

        if ch.end_of_signal:
            self.beams = right_beams

        final_beams = [IncrBeam(*beam) for beam in self.beams]
        nonfinal_beams = [IncrBeam(*beam) for beam in right_beams]

        final_transcripts = [
            LogitAlignedTranscript.create_from_token_spans(
                list(zip(b.text.split(), b.text_frames)), b.logit_score, b.lm_score
            )
            for b in islice(final_beams, self.num_best)
        ]

        nonfinal_transcripts = [
            LogitAlignedTranscript.create_from_token_spans(
                list(zip(b.text.split(), b.text_frames)), b.logit_score, b.lm_score
            )
            for b in islice(nonfinal_beams, self.num_best)
        ]

        return nonfinal_transcripts

    @beartype
    def _decode_array(
        self, beams: list[LMBeam], frame_idx: int, array: NumpyFloat2DArray
    ) -> list[LMBeam]:
        for k, logits_col in enumerate(array):
            beams = self._pyctc_decoder.incr_decode_step(
                beams, frame_idx + k, logits_col, self.params
            )
        return beams

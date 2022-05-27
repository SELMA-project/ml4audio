import itertools
from dataclasses import dataclass, field
from typing import Optional, Union, Annotated

from beartype import beartype
from beartype.vale import Is
from pyctcdecode.decoder import (
    WordFrames,
    BeamSearchDecoderCTC,
    build_ctcdecoder,
    LMState,
)

from ml4audio.text_processing.common import (
    LogitAlignedTranscript,
    BaseCTCDecoder,
    AlignedBeams,
)
from ml4audio.text_processing.lm_model_for_pyctcdecode import KenLMForPyCTCDecode
from data_io.readwrite_files import read_lines
from misc_utils.beartypes import NeList, TorchTensor2D
from misc_utils.dataclass_utils import (
    UNDEFINED,
    _UNDEFINED,
)
import kenlm
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer

LmModelFile = Annotated[
    str, Is[lambda s: any(s.endswith(suffix) for suffix in [".bin", ".arpa"])]
]


@beartype
def build_unigrams_from_lexicon_file(
    lexicon_file: str, transcript_normalizer: TranscriptNormalizer
) -> NeList[str]:
    def parse_lexicon_file(l: str) -> str:
        s = "".join(l.split("\t")[1].split(" "))
        unigram = s.replace("|", "").strip(" ")
        assert " " not in unigram
        unigram = transcript_normalizer.apply(unigram)
        return unigram

    return list({parse_lexicon_file(l) for l in read_lines(lexicon_file)})


@dataclass
class OutputBeamDc:
    text: str
    last_lm_state: LMState
    text_frames: list[WordFrames]
    logit_score: float
    lm_score: float

    def __post_init__(self):
        assert self.text == " ".join([token for token, _ in self.text_frames])


@dataclass
class PyCTCKenLMDecoder(BaseCTCDecoder):
    """
    here is huggingface's decode method: https://github.com/huggingface/transformers/blob/f275e593bfeb41b31ac8a124a9314cbd6088bfd1/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L346
    # TODO: directly use pyctcdecode's batched-decoding method -> is it faster?
    """

    lm_data: Union[
        KenLMForPyCTCDecode, _UNDEFINED
    ] = UNDEFINED  # TODO: rename lm_data to lm_model
    _pyctc_decoder: Optional[BeamSearchDecoderCTC] = field(
        init=False, repr=False, default=None
    )

    def _build_self(self) -> None:
        super()._build_self()

        # TODO: use binary-kenlm model instead of arpa
        unigrams = list(read_lines(self.lm_data.unigrams_filepath))

        self._pyctc_decoder = build_ctcdecoder(
            labels=self.vocab,
            kenlm_model_path=self.lm_data.arpa_filepath,
            unigrams=unigrams,
            alpha=self.lm_weight,  # tuned on a val set
            beta=self.beta,  # tuned on a val set
            # unk_score_offset=unk_offset,
            # is_bpe=True,
        )

    @beartype
    def decode(
        self,
        ctc_matrix: TorchTensor2D,
        lm_start_state: Optional[Union["kenlm.State", list["kenlm.State"]]] = None,
    ) -> AlignedBeams:
        beams = [
            OutputBeamDc(*b)
            for b in self._pyctc_decoder.decode_beams(
                ctc_matrix.numpy(),
                beam_width=self.beam_size,
                lm_start_state=lm_start_state,
            )
        ]

        return [
            LogitAlignedTranscript.create_from_token_spans(
                b.text_frames, b.logit_score, b.lm_score
            )
            for b in itertools.islice(beams, self.num_best)
        ]


@dataclass
class PyCTCBinKenLMDecoder(BaseCTCDecoder):
    """
    TODO: refactor!
    """

    kenlm_binary_file: Union[
        str, _UNDEFINED
    ] = UNDEFINED  # TODO: rename lm_data to lm_model
    unigrams_file: Union[
        str, _UNDEFINED
    ] = UNDEFINED  # TODO: rename lm_data to lm_model

    _pyctc_decoder: Optional[BeamSearchDecoderCTC] = field(
        init=False, repr=False, default=None
    )
    # lm_data: Optional[Any] = field(init=False, repr=False, default=None)

    def _build_self(self) -> None:
        super()._build_self()

        unigrams = list(read_lines(self.unigrams_file))

        self._pyctc_decoder = build_ctcdecoder(
            labels=self.vocab,
            kenlm_model_path=self.kenlm_binary_file,
            unigrams=unigrams,
            alpha=self.lm_weight,  # tuned on a val set
            beta=self.beta,  # tuned on a val set
            unk_score_offset=-10.0,  # see jonatas russian model
            # is_bpe=True,
        )

    @beartype
    def decode(self, ctc_matrix: TorchTensor2D) -> AlignedBeams:
        beams = [
            OutputBeamDc(*b)
            for b in self._pyctc_decoder.decode_beams(
                ctc_matrix.numpy(),
                beam_width=self.beam_size,
            )
        ]

        return [
            LogitAlignedTranscript.create_from_token_spans(
                b.text_frames, b.logit_score, b.lm_score
            )
            for b in itertools.islice(beams, self.num_best)
        ]
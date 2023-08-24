import itertools
from dataclasses import dataclass, field
from typing import Optional, Union, Annotated, Any

from beartype import beartype
from beartype.vale import Is

from data_io.readwrite_files import read_lines
from misc_utils.beartypes import NeList, NumpyFloat2DArray
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import (
    UNDEFINED,
    _UNDEFINED,
)
from ml4audio.audio_utils.overlap_array_chunker import MessageChunk
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer
from ctc_decoding.ctc_decoding import (
    BaseCTCDecoder,
    AlignedBeams,
)
from ctc_decoding.huggingface_ctc_decoding import HFCTCDecoder
from ctc_decoding.logit_aligned_transcript import LogitAlignedTranscript
from ctc_decoding.lm_model_for_pyctcdecode import (
    KenLMForPyCTCDecode,
    KenLMBinaryUnigramsFile,
)
from pyctcdecode.decoder import (
    WordFrames,
    BeamSearchDecoderCTC,
    build_ctcdecoder,
    LMState,
)

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
    """
    just to bring order into pyctcdecode's mess
    """

    text: str
    last_lm_state: LMState
    text_frames: list[WordFrames]
    logit_score: float
    lm_score: float

    def __post_init__(self):
        if len(self.text) == 0:
            self.text = " "
            assert len(self.text_frames) == 0
            self.text_frames = [(" ", (0, 0))]

        assert self.text == " ".join([token for token, _ in self.text_frames])


@dataclass
class PyCTCKenLMDecoder(HFCTCDecoder):
    """
    here is huggingface's decode method: https://github.com/huggingface/transformers/blob/f275e593bfeb41b31ac8a124a9314cbd6088bfd1/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L346
    """

    lm_weight: Union[_UNDEFINED, float] = UNDEFINED
    beta: Union[_UNDEFINED, float] = UNDEFINED
    # cannot do this with beartype NeList[str] for vocab, cause it might be a CachedList
    # vocab: Union[_UNDEFINED, list[str]] = UNDEFINED

    lm_data: Union[
        KenLMForPyCTCDecode, _UNDEFINED
    ] = UNDEFINED  # TODO: rename lm_data to lm_model

    num_best: int = 1  # number of beams to return
    beam_size: int = 100

    _pyctc_decoder: Optional[BeamSearchDecoderCTC] = field(
        init=False, repr=False, default=None
    )

    def _build_self(self) -> Any:
        super()._build_self()
        # TODO: use binary-kenlm model instead of arpa
        unigrams = list(read_lines(self.lm_data.unigrams_filepath))
        if len(unigrams) < 10_000:
            print(f"{self.lm_data.name} only got {len(unigrams)} unigrams")

        print(f"{len(unigrams)=}")
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
    def ctc_decode(
        self,
        logits: NumpyFloat2DArray,
    ) -> AlignedBeams:
        beams = [
            OutputBeamDc(*b)
            for b in self._pyctc_decoder.decode_beams(
                logits,
                beam_width=self.beam_size,
            )
        ]

        return [
            LogitAlignedTranscript.create_from_token_spans(
                b.text_frames, b.logit_score, b.lm_score
            )
            for b in itertools.islice(beams, self.num_best)
        ]

    def __del__(self):
        # for __del__ vs __delete__ see: https://stackoverflow.com/questions/59508235/what-is-the-difference-between-del-and-delete
        if self._pyctc_decoder is not None:
            self._pyctc_decoder.cleanup()  # one has to manually cleanup!


@dataclass
class PyCTCBinKenLMDecoder(BaseCTCDecoder, Buildable):
    """
    TODO: refactor!
    """

    lm_weight: Union[_UNDEFINED, float] = UNDEFINED
    beta: Union[_UNDEFINED, float] = UNDEFINED
    vocab: NeList[str] = UNDEFINED

    kenlm_binary_unigrams_file: KenLMBinaryUnigramsFile = UNDEFINED
    num_best: int = 1  # number of beams to return
    beam_size: int = 100

    _pyctc_decoder: Optional[BeamSearchDecoderCTC] = field(
        init=False, repr=False, default=None
    )
    # lm_data: Optional[Any] = field(init=False, repr=False, default=None)

    def _build_self(self) -> None:

        unigrams = list(read_lines(self.kenlm_binary_unigrams_file.unigrams_filepath))

        self._pyctc_decoder = build_ctcdecoder(
            labels=self.vocab,
            kenlm_model_path=self.kenlm_binary_unigrams_file.kenlm_binary_filepath,
            unigrams=unigrams,
            alpha=self.lm_weight,  # tuned on a val set
            beta=self.beta,  # tuned on a val set
            unk_score_offset=-10.0,  # see jonatas russian model
            # is_bpe=True,
        )

    @beartype
    def decode(self, chunk: MessageChunk) -> AlignedBeams:
        beams = [
            OutputBeamDc(*b)
            for b in self._pyctc_decoder.decode_beams(
                chunk.array,
                beam_width=self.beam_size,
            )
        ]

        return [
            LogitAlignedTranscript.create_from_token_spans(
                b.text_frames, b.logit_score, b.lm_score
            )
            for b in itertools.islice(beams, self.num_best)
        ]

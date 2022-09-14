import os
import shutil
from dataclasses import dataclass
from typing import Union

from beartype import beartype
from transformers import set_seed

from misc_utils.beartypes import TorchTensor2D
from misc_utils.dataclass_utils import UNDEFINED, _UNDEFINED
from ml4audio.asr_inference.inference import ASRAudioArrayInferencer
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    ResamplingASRLogitsInferencer,
    NumpyFloatORInt16_1DArray,
)
from ml4audio.audio_utils.aligned_transcript import (
    LetterIdx,
    NeAlignedTranscript,
)
from ml4audio.text_processing.ctc_decoding import BaseCTCDecoder, LogitAlignedTranscript

DEBUG = False
counter = 0
if DEBUG:
    # debug_name= "16kHz"
    debug_name = "8kHz"
    debug_wav_dir = f"/tmp/debug_wav_{debug_name}"
    shutil.rmtree(debug_wav_dir, ignore_errors=True)
    os.makedirs(debug_wav_dir, exist_ok=True)

set_seed(42)


@dataclass
class HFASRDecodeInferencer(ASRAudioArrayInferencer):
    """
    does asr-inference WITH decoding greedy/lm-based
    TODO:
        split into logits-inferencer and decoder
        well seems huggingface's "src/transformers/pipelines/automatic_speech_recognition.py" cannot yet do streaming! just "long audio-files"

    """

    logits_inferencer: Union[
        _UNDEFINED, ResamplingASRLogitsInferencer
    ] = UNDEFINED  # order matters! first the logits_inferencer is build which builds the transcript_normalizer which is needed by decoder!
    decoder: Union[_UNDEFINED, BaseCTCDecoder] = UNDEFINED
    # _greedy_decoder: BaseCTCDecoder = volatile_state_field()

    # def _build_self(self):
    #     self._greedy_decoder: GreedyDecoder = GreedyDecoder(
    #         transcript_normalizer=tn
    #     ).build()

    @property
    def vocab(self) -> list[str]:
        return self.logits_inferencer.vocab

    @beartype
    def transcribe_audio_array(
        self, audio_array: NumpyFloatORInt16_1DArray
    ) -> NeAlignedTranscript:
        logits = self.logits_inferencer.resample_calc_logits(audio_array)
        return self.__aligned_decode(logits, len(audio_array))

    @beartype
    def __aligned_decode(
        self, logits: TorchTensor2D, audio_array_seq_len: int
    ) -> NeAlignedTranscript:
        """
        letters aligned to audio-frames

        """
        dec_out: LogitAlignedTranscript = self.decoder.decode_logits(logits.numpy())[0]

        logits_seq_len = logits.size()[0]
        audio_to_logits_ratio = audio_array_seq_len / logits_seq_len
        projected_array_index = [
            round(audio_to_logits_ratio * i) for i in dec_out.logit_ids
        ]
        letters = [
            LetterIdx(letter=l, r_idx=i)
            for l, i in zip(dec_out.text, projected_array_index)
        ]
        return NeAlignedTranscript(
            letters=letters,
            sample_rate=self.logits_inferencer.input_sample_rate,
            logits_score=dec_out.logits_score,
            lm_score=dec_out.lm_score,
        ).update_offset()
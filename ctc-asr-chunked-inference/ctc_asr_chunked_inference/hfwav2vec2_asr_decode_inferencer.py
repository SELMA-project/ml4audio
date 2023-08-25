import os
import shutil
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from beartype import beartype
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    ASRLogitsInferencer,
    NumpyFloatORInt16_1DArray,
)
from ml4audio.audio_utils.audio_io import MAX_16_BIT_PCM
from ml4audio.audio_utils.torchaudio_utils import torchaudio_resample
from transformers import set_seed

from ctc_decoding.ctc_decoding import BaseCTCDecoder
from ctc_decoding.logit_aligned_transcript import LogitAlignedTranscript
from misc_utils.beartypes import TorchTensor2D, NumpyFloat1DArray
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import UNDEFINED, _UNDEFINED
from ml4audio.audio_utils.aligned_transcript import (
    TimestampedLetters,
)

DEBUG = False
counter = 0
if DEBUG:
    # debug_name= "16kHz"
    debug_name = "8kHz"
    debug_wav_dir = f"/tmp/debug_wav_{debug_name}"
    shutil.rmtree(debug_wav_dir, ignore_errors=True)
    os.makedirs(debug_wav_dir, exist_ok=True)

set_seed(42)


@beartype
def convert_and_resample(
    audio: NumpyFloatORInt16_1DArray, input_sample_rate: int, target_sample_rate: int
) -> NumpyFloat1DArray:
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / MAX_16_BIT_PCM
    if input_sample_rate != target_sample_rate:
        audio = torchaudio_resample(
            signal=torch.from_numpy(audio.astype(np.float32)),
            sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
        ).numpy()
    return audio


@dataclass
class HFASRDecodeInferencer(Buildable):
    """
    does asr-inference WITH decoding greedy/lm-based
    TODO:
        split into logits-inferencer and decoder
        well seems huggingface's "src/transformers/pipelines/automatic_speech_recognition.py" cannot yet do streaming! just "long audio-files"

    """

    input_sample_rate: int = 16000
    logits_inferencer: ASRLogitsInferencer = UNDEFINED  # order matters! first the logits_inferencer is build which builds the transcript_normalizer which is needed by decoder!
    decoder: Union[_UNDEFINED, BaseCTCDecoder] = UNDEFINED

    @property
    def vocab(self) -> list[str]:
        return self.logits_inferencer.vocab

    @beartype
    def transcribe_audio_array(
        self, audio_array: NumpyFloatORInt16_1DArray
    ) -> TimestampedLetters:
        audio_array = convert_and_resample(
            audio_array,
            self.input_sample_rate,
            self.logits_inferencer.asr_model_sample_rate,
        )
        logits = self.logits_inferencer.calc_logits(audio_array)
        return self.__aligned_decode(logits, len(audio_array))

    @beartype
    def __aligned_decode(
        self, logits: TorchTensor2D, audio_array_seq_len: int
    ) -> TimestampedLetters:
        """
        letters aligned to audio-frames

        """
        dec_out: LogitAlignedTranscript = self.decoder.ctc_decode(logits.numpy())[0]

        logits_seq_len = logits.size()[0]
        audio_to_logits_ratio = audio_array_seq_len / logits_seq_len
        timestamps = [
            audio_to_logits_ratio * i / self.input_sample_rate
            for i in dec_out.logit_ids
        ]

        return TimestampedLetters(
            dec_out.text, np.array(timestamps)
        )  # ,dec_out.logits_score,dec_out.lm_score

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import whisper as whisper_module
from beartype import beartype
from whisper import Whisper, DecodingOptions

from misc_utils.beartypes import NumpyFloat1DArray, NumpyFloat1D
from misc_utils.dataclass_utils import UNDEFINED, FillUndefined
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.asr_inference.inference import (
    StartEndTextsNonOverlap,
)
from ml4audio.asr_inference.whisper_inference import (
    WhisperInferencer,
    fix_whisper_segments,
    WhisperArgs,
)
from whisper.utils import exact_div


@dataclass(frozen=True)
class OpenAiWhisperArgs(WhisperArgs, DecodingOptions):
    """
    for defaults see transcribe-method
    """

    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"



@dataclass
class OpenAIWhisperASRSegmentInferencer(WhisperInferencer):
    """
    https://github.com/saharmor/whisper-playground
    """

    model_name: str = "base"
    whisper_args: Optional[OpenAiWhisperArgs] = None
    _model: Whisper = field(init=False, repr=False)
    base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "MODELS/WHISPER_MODELS"),
        init=False,
    )

    def __post_init__(self):
        if self.model_name.startswith("openai/whisper-"):
            self.model_name = self.model_name.replace("openai/whisper-", "")

    @property
    def name(self) -> str:
        return f"whisper-{self.model_name}"

    @property
    def sample_rate(self) -> int:
        return whisper_module.audio.SAMPLE_RATE

    @property
    def _is_data_valid(self) -> bool:
        return os.path.isfile(self._checkpoint_file)

    @property
    def _checkpoint_file(self) -> str:
        """
        see: whisper/__init__.py _download method
        """
        return f"{self.data_dir}/{os.path.basename(whisper_module._MODELS[self.model_name])}"

    def _build_data(self) -> Any:
        checkpoint_file = whisper_module._download(
            whisper_module._MODELS[self.model_name], self.data_dir, in_memory=False
        )
        assert checkpoint_file == self._checkpoint_file

    def __enter__(self):
        self._model = whisper_module.load_model(self._checkpoint_file)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        del self._model

    @beartype
    def predict_transcribed_with_whisper_args(
        self, audio_array: NumpyFloat1D, whisper_args: OpenAiWhisperArgs
    ):  # -> StartEndTextsNonOverlap:
        from whisper import audio

        audio.CHUNK_LENGTH = whisper_args.chunk_length
        audio.N_SAMPLES = audio.CHUNK_LENGTH * audio.SAMPLE_RATE
        audio.N_FRAMES = exact_div(audio.N_SAMPLES, audio.HOP_LENGTH)

        audio_dur = float(len(audio_array) / self.sample_rate)
        resp = self._model.transcribe(audio=audio_array,**asdict(whisper_args))

        # resp["text"].strip(" ") # somehow this sometimes repeats the transcribt twice
        whisper_segments = resp["segments"]
        if len(whisper_segments) > 0:
            raw_whisper_segments = [
                (seg["start"], seg["end"], seg["text"]) for seg in whisper_segments
            ]
            start_end_text = fix_whisper_segments(
                raw_whisper_segments,
                audio_dur,
            )
        else:
            start_end_text = []
        return start_end_text


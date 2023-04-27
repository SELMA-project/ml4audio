import os
from dataclasses import dataclass, field, asdict
from typing import Any, Annotated, Optional, Union

from beartype import beartype
from beartype.vale import Is

import whisper as whisper_module
from misc_utils.beartypes import NumpyFloat1DArray, NumpyFloat1D
from misc_utils.dataclass_utils import UNDEFINED, FillUndefined
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.asr_inference.inference import (
    StartEndTextsNonOverlap,
)
from ml4audio.asr_inference.whisper_inference import (
    WhisperArgs,
    WhisperInferencer,
    fix_whisper_segments, WHISPER_TASKS,
)
from whisper import Whisper, DecodingOptions


@dataclass(frozen=True)
class OpenAiWhisperArgs(DecodingOptions):
    task: Annotated[str, Is[lambda s: s in WHISPER_TASKS]]="transcribe"
    language: str = "de"
    temperature: Optional[Union[float, tuple[float, ...], list[float]]] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )  # this is default in whisper code
    # don't mess with the temperatures! they are needed for fallback if beam-search fails!
    beam_size: Optional[int] = None  # default=5 see whisper code

    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"

@dataclass(frozen=True)
class WhisperPredictArgs(OpenAiWhisperArgs, FillUndefined):
    audio: NumpyFloat1DArray = UNDEFINED


@dataclass
class OpenAIWhisperASRSegmentInferencer(WhisperInferencer):
    """
    https://github.com/saharmor/whisper-playground
    """

    model_name: str = "base"
    whisper_args: Optional[OpenAiWhisperArgs] = None
    _model: Whisper = field(init=False, repr=False)
    base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "MODELS/WHISPER_MODELS")
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
    ) -> StartEndTextsNonOverlap:
        audio_dur = float(len(audio_array) / self.sample_rate)
        pred_args = WhisperPredictArgs(audio=audio_array, **asdict(whisper_args))
        resp = self._model.transcribe(**asdict(pred_args))

        # resp["text"].strip(" ") # somehow this sometimes repeats the transcribt twice
        whisper_segments = resp["segments"]
        if len(whisper_segments) > 0:
            start_end_text = fix_whisper_segments(
                [(seg["start"], seg["end"], seg["text"]) for seg in whisper_segments],
                audio_dur,
            )
        else:
            start_end_text = []
        return start_end_text


if __name__ == "__main__":
    base_path = os.environ.get("BASE_PATH", "/tmp")
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["cache_root"] = cache_root
    prompt_extlm="die Schule, die Kindertagesstätte, die Kita "
    inferencer = OpenAIWhisperASRSegmentInferencer(
        model_name="base",
        whisper_args=OpenAiWhisperArgs(
            task="transcribe", language="de",
            temperature=0.0,
            beam_size=5,
            external_lm_model_name="dbmdz/german-gpt2",
            prompt_for_extlm=prompt_extlm
        ),
    )
    inferencer.build()
    from ml4audio.audio_utils.audio_io import ffmpeg_load_trim

    file = "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"
    array = ffmpeg_load_trim(file)
    with inferencer:
        print(f"{inferencer.predict_transcribed_segments(array)=}")

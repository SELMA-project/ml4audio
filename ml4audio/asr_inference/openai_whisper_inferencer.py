import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from beartype import beartype

import whisper as whisper_module
from misc_utils.beartypes import NumpyFloat1DArray, NeList, NumpyFloat1D
from misc_utils.buildable_data import BuildableData
from misc_utils.dataclass_utils import UNDEFINED, FillUndefined
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.asr_inference.inference import (
    ASRAudioSegmentInferencer,
    StartEndTextsNonOverlap,
)
from ml4audio.asr_inference.whisper_inference import (
    fix_start_end,
    WhisperArgs,
    WhisperInferencer,
)
from ml4audio.audio_utils.audio_segmentation_utils import (
    fix_segments_to_non_overlapping,
)
from whisper import Whisper


@dataclass
class WhisperPredictArgs(WhisperArgs, FillUndefined):
    audio: NumpyFloat1DArray = UNDEFINED


@dataclass
class OpenAIWhisperASRSegmentInferencer(BuildableData, WhisperInferencer):
    """
    https://github.com/saharmor/whisper-playground
    """

    model_name: str = "base"

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
    def parse_whisper_segments(
        self, whisper_segments: NeList[dict], audio_dur: float
    ) -> StartEndTextsNonOverlap:

        start_end = [fix_start_end(seg, audio_dur) for seg in whisper_segments]
        start_end = fix_segments_to_non_overlapping(start_end)
        return [
            (start, end, seg["text"])
            for seg, (start, end) in zip(whisper_segments, start_end)
        ]


    @beartype
    def predict_transcribed_with_whisper_args(
        self, audio_array: NumpyFloat1D, whisper_args: WhisperArgs
    ) -> StartEndTextsNonOverlap:
        audio_dur = float(len(audio_array) / self.sample_rate)
        pred_args = WhisperPredictArgs(audio=audio_array, **asdict(whisper_args))
        resp = self._model.transcribe(**asdict(pred_args))

        # resp["text"].strip(" ") # somehow this sometimes repeats the transcribt twice
        whisper_segments = resp["segments"]
        if len(whisper_segments) > 0:
            start_end_text = self.parse_whisper_segments(whisper_segments, audio_dur)
        else:
            start_end_text = []
        return start_end_text


if __name__ == "__main__":
    base_path = os.environ.get("BASE_PATH", "/tmp")
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["cache_root"] = cache_root

    inferencer = OpenAIWhisperASRSegmentInferencer(
        model_name="base", whisper_args=WhisperArgs(task="transcribe", language="en")
    )
    inferencer.build()
    from ml4audio.audio_utils.audio_io import ffmpeg_load_trim

    file = "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"
    array = ffmpeg_load_trim(file)
    with inferencer:
        print(f"{inferencer.predict_transcribed_segments(array)=}")

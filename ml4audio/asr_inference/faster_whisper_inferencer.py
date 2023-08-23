from dataclasses import dataclass, field
from typing import Any, Optional

import faster_whisper
from beartype import beartype
from faster_whisper import download_model, WhisperModel

import whisper as whisper_module
from misc_utils.beartypes import NumpyFloat1D
from misc_utils.prefix_suffix import PrefixSuffix
from ml4audio.asr_inference.whisper_inference import (
    WhisperInferencer,
    WhisperArgs,
)


@dataclass
class FasterWhisperASRSegmentInferencer(WhisperInferencer):

    model_name: str = "base"
    whisper_args: Optional[WhisperArgs] = None
    num_threads: int = 4
    _model: WhisperModel = field(init=False, repr=False)
    base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "MODELS/WHISPER_MODELS"),
        init=False,
    )

    @property
    def name(self) -> str:
        return f"faster-whisper-{self.model_name}"

    @property
    def sample_rate(self) -> int:
        return whisper_module.audio.SAMPLE_RATE

    @property
    def _is_data_valid(self) -> bool:
        try:
            download_model(
                self.model_name,
                local_files_only=True,
                cache_dir=self.data_dir,
            )
            found_it = True
        except FileNotFoundError:
            found_it = False
        return found_it

    def _build_data(self) -> Any:
        model_path = download_model(
            self.model_name,
            local_files_only=False,
            cache_dir=self.data_dir,
        )

    def __enter__(self):
        model_path = download_model(
            self.model_name,
            local_files_only=True,
            cache_dir=self.data_dir,
        )
        self._model = faster_whisper.WhisperModel(
            model_path, device="cpu", compute_type="int8", cpu_threads=self.num_threads
        )

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        del self._model

    @beartype
    def predict_transcribed_with_whisper_args(
        self, audio_array: NumpyFloat1D, whisper_args: WhisperArgs
    ):  # -> StartEndTextsNonOverlap:
        segments, _ = self._model.transcribe(
            audio_array,
            beam_size=5,
            word_timestamps=True,
            # initial_prompt="Welcome to the software engineering courses channel.",
        )
        segments = list(segments)
        return [(seg.start, seg.end, seg.text) for seg in segments]

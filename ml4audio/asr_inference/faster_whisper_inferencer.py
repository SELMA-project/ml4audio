from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union, Iterable

import faster_whisper
from beartype import beartype
from faster_whisper import download_model, WhisperModel
from faster_whisper.vad import VadOptions

import whisper as whisper_module
from misc_utils.beartypes import NumpyFloat1D
from misc_utils.buildable_data import BuildableData
from misc_utils.prefix_suffix import PrefixSuffix
from ml4audio.asr_inference.inference import (
    StartEndTextsNonOverlap,
    AudioArray2SegmentedTranscripts,
)
from ml4audio.asr_inference.whisper_inference import (
    WhisperArgs,
    fix_whisper_segments,
)


@dataclass
class FasterWhisperArgs:
    """
    this is based on the faster-whispers transcribe-methods input arguments
    see: https://github.com/guillaumekln/faster-whisper/blob/7b271da0351e4f81f80e8bb4d2c21c9406475aa9/faster_whisper/transcribe.py#L155
    """

    # audio: Union[str, BinaryIO, np.ndarray],
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    temperature: Union[float, list[float], tuple[float, ...]] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )
    compression_ratio_threshold: Optional[float] = 2.4
    log_prob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    condition_on_previous_text: bool = True
    # prompt_reset_on_temperature: float = 0.5
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    prefix: Optional[str] = None
    suppress_blank: bool = True
    suppress_tokens: Optional[list[int]] = field(default_factory=lambda: [-1])
    without_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"
    vad_filter: bool = False
    vad_parameters: Optional[Union[dict, VadOptions]] = None


@dataclass
class FasterWhisperArray2SegmentedTranscripts(
    BuildableData, AudioArray2SegmentedTranscripts
):

    model_name: str = "base"
    whisper_args: FasterWhisperArgs = None
    num_threads: int = 4
    compute_type: str = "int8"  # (possible values are: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16, float32)
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
            model_path,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.num_threads,
        )

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        del self._model

    @beartype
    def audio_to_segmented_transcripts(
        self, audio_array: NumpyFloat1D
    ) -> StartEndTextsNonOverlap:
        return self.predict_transcribed_with_whisper_args(
            audio_array, self.whisper_args
        )

    @beartype
    def predict_transcribed_with_whisper_args(
        self, audio_array: NumpyFloat1D, whisper_args: FasterWhisperArgs
    ) -> StartEndTextsNonOverlap:
        segments, _ = self._model.transcribe(audio_array, **asdict(whisper_args))
        segments = list(segments)
        if len(segments) > 0:
            raw_whisper_segments = [(seg.start, seg.end, seg.text) for seg in segments]
            audio_dur = float(len(audio_array) / self.sample_rate)
            start_end_text = fix_whisper_segments(
                raw_whisper_segments,
                audio_dur,
            )
        else:
            start_end_text = []
        return start_end_text

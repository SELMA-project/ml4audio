import os
from dataclasses import dataclass, field, asdict
from typing import Iterator, Any, Union, Optional, Annotated

from beartype import beartype
from beartype.vale import Is

import whisper as whisper_module
from misc_utils.beartypes import NumpyFloat1DArray
from misc_utils.buildable_data import BuildableData
from misc_utils.dataclass_utils import UNDEFINED, FillUndefined
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from whisper import Whisper

WHISPER_TASKS = {"transcribe", "translate"}


@dataclass
class WhisperArgs:
    task: Annotated[str, Is[lambda s: s in WHISPER_TASKS]]
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


@dataclass
class WhisperPredictArgs(WhisperArgs, FillUndefined):
    audio: NumpyFloat1DArray = UNDEFINED


@dataclass
class WhisperInferencer(BuildableData):
    """
    https://github.com/saharmor/whisper-playground
    """

    model_name: str = "base"

    _model: Whisper = field(init=False, repr=False)
    base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "MODELS/WHISPER_MODELS")
    )

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
        self._load_data()

    def _load_data(self):
        self._model = whisper_module.load_model(self._checkpoint_file)

    @beartype
    def predict(self, pred_args: WhisperPredictArgs) -> dict:

        result = self._model.transcribe(**asdict(pred_args))
        # TODO: result as dataclass?
        return result


if __name__ == "__main__":
    base_path = os.environ.get("BASE_PATH", "/tmp")
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["cache_root"] = cache_root

    inferencer = WhisperInferencer(model_name="tiny")
    inferencer.build()
    from ml4audio.audio_utils.audio_io import ffmpeg_load_trim

    file = "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"
    array = ffmpeg_load_trim(file)
    print(f"{inferencer.predict(array)=}")

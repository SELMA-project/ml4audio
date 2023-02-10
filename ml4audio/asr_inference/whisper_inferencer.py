import os
from dataclasses import dataclass, field
from typing import Iterator, Any, Union, Optional

from beartype import beartype

import whisper as whisper_module
from misc_utils.beartypes import NumpyFloat1DArray
from misc_utils.buildable_data import BuildableData
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.audio_utils.audio_data_models import IdText
from whisper import Whisper


@dataclass
class WhisperInferencer(BuildableData):
    model_name: str = "base"
    lang: str = "de"

    temperature: Optional[Union[float, tuple[float, ...], list[float]]] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )  # this is default in whisper code
    beam_size: Optional[int] = None
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
    def predict(self, audio_array: NumpyFloat1DArray) -> str:

        result = self._model.transcribe(
            audio_array,
            task="transcribe",
            language=self.lang,
            beam_size=self.beam_size,
            temperature=self.temperature,  # don't mess with the temperatures! they are needed for fallback if beam-search fails!
        )
        return result["text"].strip(" ")


if __name__ == "__main__":
    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["cache_root"] = cache_root

    whisper = WhisperInferencer(model_name="tiny", lang="de")
    whisper.build()

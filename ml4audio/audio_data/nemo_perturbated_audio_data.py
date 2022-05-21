import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Iterator,
    Optional,
    Union,
)

import soundfile as sf
import torch
from beartype import beartype
from nemo.collections.asr.parts.preprocessing import (
    AudioAugmentor,
)
from tqdm import tqdm

from ml4audio.audio_utils.audio_data_models import AudioData, IdArray
from ml4audio.audio_data.nemo_perturbation import (
    apply_nemo_perturbations_with_retry,
    ProbaPerturbationDC,
)
from ml4audio.audio_utils.audio_io import (
    normalize_audio_array,
)
from ml4audio.audio_utils.torchaudio_utils import torchaudio_resample, torchaudio_load
from misc_utils.beartypes import (
    NumpyFloat1DArray,
)
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import (
    _UNDEFINED,
    UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix


@dataclass
class NemoPerturbatedAudioData(CachedData, AudioData):
    """
    for benchmarking not for training
    """

    raw_data: Union[_UNDEFINED, AudioData] = UNDEFINED
    perturbations: Optional[list[ProbaPerturbationDC]] = None
    perturbation_name: Union[_UNDEFINED, str] = UNDEFINED
    augmentor: AudioAugmentor = field(init=False, repr=False)

    cache_base: PrefixSuffix = field(
        default_factory=lambda: BASE_PATHES["processed_data"]
    )

    @property
    def name(self) -> str:
        return f"{self.raw_data.name}-{self.perturbation_name}"

    def _build_cache(self) -> None:
        os.makedirs(self.prefix_cache_dir(f"wavs"), exist_ok=False)
        proba_perts = [(pb.proba, pb) for pb in self.perturbations]

        self.augmentor = (
            AudioAugmentor(perturbations=proba_perts)
            if self.perturbations is not None and len(self.perturbations) > 0
            else None
        )
        for id_array in tqdm(
            self.raw_data, desc=f"augmenting/perturbating {self.raw_data.name}"
        ):
            self.process(id_array)

    @beartype
    def process(self, id_array: IdArray):
        eid, array = id_array
        processed_audio_file = self.prefix_cache_dir(f"wavs/{eid}.wav")
        array = normalize_audio_array(array)
        array = apply_nemo_perturbations_with_retry(
            array, sample_rate=self.raw_data.sample_rate, augmentor=self.augmentor
        )
        if self.sample_rate != self.raw_data.sample_rate:
            array = torchaudio_resample(
                torch.from_numpy(array),
                self.raw_data.sample_rate,
                self.sample_rate,
            ).numpy()

        sf.write(
            processed_audio_file,
            array,
            samplerate=self.sample_rate,  # see SoxPerturbations
        )

    def __iter__(self) -> Iterator[tuple[str, NumpyFloat1DArray]]:
        for p in Path(self.prefix_cache_dir("wavs")).glob("*.wav"):
            array, _ = torchaudio_load(str(p))
            eid = p.name.replace(".wav", "")
            yield eid, array.numpy()

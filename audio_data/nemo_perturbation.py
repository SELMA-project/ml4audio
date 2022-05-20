import os
import subprocess
import traceback
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
from typing import Any, Union, Optional

import numpy as np
import soundfile as sf
from beartype import beartype
from nemo.collections.asr.parts.preprocessing import (
    Perturbation,
    AudioSegment,
    TranscodePerturbation,
    AudioAugmentor,
)

from audio_utils.audio_io import normalize_audio_array
from misc_utils.beartypes import NumpyFloat1DArray
from misc_utils.dataclass_utils import UNDEFINED, _UNDEFINED
from audio_data.sox_signal_augmentation import (
    add_signals,
    build_sox_distortions_piped,
    build_random_bandpass_cutoffs,
    build_dynamic_noise,
)
from audio_data.sox_signal_augmentation import build_random_noise
from audio_data.sox_signal_augmentation import varying_gain_pert

@dataclass
class ProbaPerturbationDC(Perturbation):
    proba: float = 1.0


@dataclass
class TranscodePerturbationDC(TranscodePerturbation, ProbaPerturbationDC):
    def __post_init__(self):
        super().__init__(None)


@dataclass
class SampleRatePerturbationDC(ProbaPerturbationDC):
    sample_rate: int = 16000


@dataclass
class SoxPerturbations(ProbaPerturbationDC):
    """
     see: http://sox.sourceforge.net/sox.html
        pert_params = {
        "tempo": round(np.random.triangular(left=0.8, mode=1.0, right=1.2), 2),
        "pitch": int(round(np.random.triangular(left=-150, mode=0, right=150))),
        # normal 100, less: 50, evenless: 30
        "reverb": (int(round(np.random.uniform(low=0, high=50))), 50, 100, 100, 0, 0,),
        "gain -n": signal_gain,
    }

    """

    pert_params: Union[_UNDEFINED, dict[str, Any]] = field(init=True, default=UNDEFINED)

    def max_augmentation_length(self, length):
        return (
            1.3 * length
        )  # rought estimate -> wtf! where does this estimate come from?

    def perturb(self, data: AudioSegment):
        norm_samples = normalize_audio_array(data)
        with NamedTemporaryFile(
            suffix=".wav", delete=True
        ) as orig_f, NamedTemporaryFile(
            suffix="_augmented.wav", delete=True
        ) as tmp_file:
            sr = data.sample_rate
            sf.write(orig_f.name, norm_samples.transpose(), sr)

            original = orig_f.name
            augmented = tmp_file.name

            sox_cmd = self._build_sox_command(augmented, original)
            # print(f"{sox_cmd=}")
            FNULL = open(os.devnull, "w")
            subprocess.call(
                ["bash", "-c", sox_cmd, "> /dev/null 2>&1"],
                stdout=FNULL,
                stderr=subprocess.STDOUT,
            )
            try:
                new_data = AudioSegment.from_file(augmented, target_sr=16000)
            except Exception as e:
                print(f"this one failed: {sox_cmd=}")
                raise e

        data._samples = new_data._samples

    def _build_sox_command(self, augmented_file: str, original_file: str) -> str:
        sox_pipe = build_sox_distortions_piped(original_file, self.pert_params)
        return f"sox <({sox_pipe}) -b 16 {augmented_file}"


def default_create_sox_cmd_fun(
    augmented_file: str,
    original_file: str,
) -> str:
    min_SNR = 20.0
    signal_gain = round(
        np.random.triangular(left=-10, mode=0.0, right=30), 2
    )  # to provoke clipping!
    noise = build_random_noise(min_SNR, original_file, signal_gain)
    gain_pert_sig = varying_gain_pert(original_file)
    lowpass, highpass = build_random_bandpass_cutoffs(
        min_low=1000, min_band_width=100, max_high=1000
    )
    if lowpass is None and highpass is None:
        sinc_params = "1"  # highpass of very low freq (1) defacto "allpass"
    elif lowpass is None and highpass is not None:
        sinc_params = f"{highpass}"
    elif highpass is None and lowpass is not None:
        sinc_params = f"-{lowpass}"
    else:
        sinc_params = f"{highpass}-{lowpass}"
    # print(f"{sinc_params=}")
    # fmt:off
    pert_params = {
        "tempo": round(np.random.triangular(left=0.8, mode=1.0, right=1.2), 2),
        "pitch": int(round(np.random.triangular(left=-150, mode=0, right=150))),
        # normal 100, less: 50, evenless: 30
        "reverb": (
            int(round(np.random.uniform(low=0, high=50))), 50, 100, 100, 0, 0,),
        "gain -n": signal_gain,
        "sinc": sinc_params
    }
    # fmt:on

    pert_sig = build_sox_distortions_piped(gain_pert_sig, pert_params)

    sox_cmd = add_signals([noise, pert_sig], augmented_file)
    return sox_cmd


@dataclass
class ManyRandomSoxPerturbations(SoxPerturbations):
    # create_sox_cmd_fun: Optional[ Callable[[str, str], SOX_CMD]] = None  # TODO: beartype complains
    # cannot hand in method as argument! -> this is against the rules!

    pert_params: Union[_UNDEFINED, dict[str, Any]] = field(
        init=False, default=UNDEFINED, repr=False
    )

    def _build_sox_command(self, augmented_file, original_file) -> str:
        return default_create_sox_cmd_fun(augmented_file, original_file)


@dataclass
class BandPassPerturb(SoxPerturbations):
    lowpass: int = None
    highpass: int = None

    def _build_sox_command(self, augmented_file, original_file) -> str:
        assert self.highpass < self.lowpass
        pert_params = {
            "bandpass": [self.highpass, self.lowpass]
        }  # first high than low-pass!!
        sox_pipe = build_sox_distortions_piped(original_file, pert_params)
        return f"sox <({sox_pipe}) -b 16 {augmented_file}"


@dataclass
class PitchPerturb(SoxPerturbations):
    pitch: int = None

    def _build_sox_command(self, augmented_file, original_file) -> str:
        pert_params = {"pitch": self.pitch}
        sox_pipe = build_sox_distortions_piped(original_file, pert_params)
        return f"sox <({sox_pipe}) -b 16 {augmented_file}"


@dataclass
class SingleSox(SoxPerturbations):
    pitch: int = None

    def _build_sox_command(self, augmented_file, original_file) -> str:
        pert_params = {"pitch": self.pitch}
        sox_pipe = build_sox_distortions_piped(original_file, pert_params)
        return f"sox <({sox_pipe}) -b 16 {augmented_file}"




@beartype
def apply_nemo_perturbations_with_retry(
    audio_array: NumpyFloat1DArray,
    sample_rate: int,
    augmentor: Optional[AudioAugmentor] = None,
    max_retries: int = 3,
) -> NumpyFloat1DArray:
    audio = AudioSegment(samples=audio_array, sample_rate=sample_rate)
    for k in range(1, max_retries + 1):
        try:
            if augmentor is not None:
                augmentor.perturb(audio)
            break
        except Exception as e:
            traceback.print_exc()
            print(f"{e=}")
            if k > 0:
                print(f"pertubration retry {k} of {max_retries} failed!")
    return audio.samples


if __name__ == "__main__":
    original_file = (
        "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.wav"
    )
    # noise = build_random_noise(10, original_file, 0.0)
    noise = build_dynamic_noise(original_file, lowpass_cutoff=4000, highpass_cutoff=100)
    perturbed = "/tmp/noise.wav"
    sox_cmd = f"sox <({noise}) -b 16 {perturbed}"
    print(f"{sox_cmd=}")
    FNULL = open(os.devnull, "w")
    subprocess.call(
        ["bash", "-c", sox_cmd, "> /dev/null 2>&1"],
        stdout=FNULL,
        stderr=subprocess.STDOUT,
    )

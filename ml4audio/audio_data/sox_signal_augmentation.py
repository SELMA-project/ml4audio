import os
import random
import subprocess
from typing import Optional

import numpy as np
from beartype import beartype

MAX_FREQ = 7000  # not 79999 (MAX_FREQ), sox complains: "sox FAIL sinc: filter frequency must be less than sample-rate / 2"

SOX_CMD = str


def to_str(v):
    if isinstance(v, (tuple, list)):
        s = " ".join(str(x) for x in v)
    elif isinstance(v, float) or isinstance(v, int):
        s = str(v)
    elif isinstance(v, str):
        s = v
    else:
        assert False

    return s


def transcode_perturbation(file, output_file):
    """
    stolen from nvidia/nemo
    """
    _rng = np.random.RandomState()
    _codecs = ["g711", "amr-nb"]

    codec_ind = random.randint(0, len(_codecs) - 1)
    if _codecs[codec_ind] == "amr-nb":
        rates = list(range(0, 8))
        rate = rates[random.randint(0, len(rates) - 1)]
        _ = subprocess.check_output(
            f"sox {file} -V0 -C {rate} -t amr-nb - | sox -t amr-nb - -V0 -b 16 -r 16000 {output_file}",
            shell=True,
        )
    elif _codecs[codec_ind] == "g711":
        _ = subprocess.check_output(
            f"sox {file} -V0  -r 8000 -c 1 -e a-law {output_file}", shell=True
        )


@beartype
def build_sox_distortions_piped(piped_or_file: str, params: dict) -> SOX_CMD:
    if not piped_or_file.startswith("sox"):
        assert os.path.isfile(piped_or_file)
    else:
        piped_or_file = f"<({piped_or_file})"

    param_str = " ".join([k + " " + to_str(v) for k, v in params.items()])
    sox_params = f"sox {piped_or_file} -p {param_str}"
    return sox_params


@beartype
def build_dynamic_noise(
    audio_file: str,
    amod_lowpass_cutoff: float = 0.1,
    lowpass_cutoff: int = MAX_FREQ,  # max noise freq
    highpass_cutoff: int = 1,  # min noise freq
    noise_gain: float = -4,
):
    """
        band-pass-filtered whitenoise multiplied by very-low-freq whitenoise
        gives non-static/dynamically chaning noise
        :param amod_lowpass_cutoff: upper freq for noise-power changes, how "dynamic" noise is

    play tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.wav synth whitenoise lowpass 0.1 synth whitenoise amod gain -n 0 sinc 1-1000
    """

    sox_params = (
        f"sox {audio_file} -p synth whitenoise lowpass {amod_lowpass_cutoff} "
        f"synth whitenoise amod gain -n {noise_gain} sinc {highpass_cutoff}-{lowpass_cutoff}"
    )
    return sox_params


def build_varying_amplitude_factor(audio_file, lowpass_cutoff=1, ac_gain=-9):
    """
    lowpass_cutoff is upper freq of ac component
    """
    ac = f"sox {audio_file} -p synth whitenoise lowpass {lowpass_cutoff} gain -n {ac_gain}"
    # WTF! dc is made by muting the original-signal and giving it an offset/dcshift!! why 0.5??
    dc = f"sox {audio_file} -p gain -90 dcshift 0.5"
    return f"sox -m <({ac}) <({dc}) -p"


@beartype
def multiply_signals(signal_a: str, signal_b: str) -> SOX_CMD:
    return f"sox -T <({signal_a}) <({signal_b}) -p"


@beartype
def varying_gain_pert(audio_file, upper_freq_for_gain_var=1, ac_gain=-6) -> SOX_CMD:
    factor = build_varying_amplitude_factor(
        audio_file, upper_freq_for_gain_var, ac_gain
    )
    signal = f"sox {audio_file} -p "
    return multiply_signals(factor, signal)


def add_signals_trim_to_len(original, signals, augmented):
    signals_to_add = " ".join([f"<({s})" for s in signals])
    sox_cmd = f"sox -m {signals_to_add} -b 16 {augmented} trim 0 $(soxi -D {original})"
    return sox_cmd


def add_signals(signals, outfile):
    signals_to_add = " ".join([f"<({s})" for s in signals])
    sox_cmd = f"sox -m {signals_to_add} -b 16 {outfile}"
    return sox_cmd


def log_uniform(low, high):
    return np.exp(np.random.uniform(low=np.log(low), high=np.log(high)))


@beartype
def build_random_bandpass_cutoffs(
    min_low=101, min_band_width=100, max_high=1000
) -> tuple[Optional[int], Optional[int]]:
    """
    :param min_low: minimal low-pass-freq, in "worst-case" cut away everything above min_low
    :param min_band_width:
    :param max_high: maximal hight-pass freq, in "worst-case" cuts away everything below this
    :return:
    """
    assert min_low - min_band_width > 0
    max_high_cutoff = MAX_FREQ
    if np.random.choice([True, False], p=[0.5, 0.5]):
        lowpass = int(round(log_uniform(low=min_low, high=MAX_FREQ)))
        max_high_cutoff = lowpass - min_band_width
    else:
        lowpass = None

    if np.random.choice([True, False], p=[0.5, 0.5]):
        highpass = int(round(log_uniform(low=2, high=min(max_high, max_high_cutoff))))
    else:
        highpass = None

    return lowpass, highpass


def build_random_noise(min_SNR: float, original_file, signal_gain) -> str:
    noise_power = round(np.random.uniform(-60, signal_gain - min_SNR), 2)
    lowpass = int(round(np.random.uniform(low=1000, high=MAX_FREQ)))
    highpass = int(round(np.random.uniform(low=1, high=lowpass)))
    noise = build_dynamic_noise(
        original_file,
        np.random.uniform(0.1, 2),
        lowpass,
        highpass,
        noise_gain=noise_power,
    )
    return noise


def build_random_pert(sig, signal_gain=0.0) -> SOX_CMD:
    # fmt:off
    pert_params = {
        "tempo": round(np.random.triangular(left=0.8, mode=1.0, right=1.2), 2),
        "pitch": int(round(np.random.triangular(left=-150, mode=0, right=150))),
        # normal 100, less: 50, evenless: 30
        "reverb": (int(round(np.random.uniform(low=0, high=50))), 50, 100, 100, 0, 0,),
        "gain -n": signal_gain,
    }
    # fmt:on
    lowpass, highpass = build_random_bandpass_cutoffs(200, 100, 1000)

    pert_params["sinc"] = f"{highpass}-{lowpass}"
    sox_cmd = build_sox_distortions_piped(sig, pert_params)
    return sox_cmd

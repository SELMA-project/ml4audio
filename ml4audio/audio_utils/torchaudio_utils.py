from io import BytesIO
from tarfile import ExFileObject
from typing import Any, Optional, Union

import numpy as np
import torch
import torchaudio
from beartype import beartype
from torch import float32
from torchaudio.transforms import Resample

from misc_utils.beartypes import (
    TorchTensor1D,
)

resamplers: dict[str, Resample] = {}


@beartype
def get_resampler(
    sample_rate: int, target_sample_rate: int, dtype: Optional[torch.dtype]
) -> Resample:
    """
    could one use a singleton here?
    """
    global resamplers
    key = f"{sample_rate}-{target_sample_rate}-{dtype}"
    if key not in resamplers:
        resamplers[key] = Resample(sample_rate, target_sample_rate, dtype=dtype)
    return resamplers[key]


@beartype
def torchaudio_resample(
    signal: TorchTensor1D, sample_rate: int, target_sample_rate: int
) -> TorchTensor1D:
    if target_sample_rate != sample_rate:
        resampler = get_resampler(
            sample_rate, target_sample_rate=target_sample_rate, dtype=float32
        )
        signal = resampler(signal)
    return signal


@beartype
def torchaudio_info(audio_file: str) -> tuple[int, int, float]:
    info = torchaudio.info(audio_file)  # torchaudio==0.8.0
    num_frames = info.num_frames
    sample_rate = info.sample_rate
    duration = num_frames / sample_rate
    return num_frames, sample_rate, duration


@beartype
def _parse_offset_for_torchaudio_load(
    offset: Optional[Union[int, float]], sample_rate: Optional[int] = None
) -> int:
    if offset is None:
        frame_offset = 0
    elif isinstance(offset, float):
        assert sample_rate is not None
        frame_offset = round(offset * sample_rate)
    else:
        frame_offset = offset
    return frame_offset


@beartype
def _parse_duration_for_torchaudio_load(
    duration: Optional[Union[int, float]], sample_rate: Optional[int] = None
) -> int:
    if duration is None:
        num_frames = -1
    elif isinstance(duration, float):
        assert sample_rate is not None
        num_frames = round(duration * sample_rate)
    elif isinstance(duration, int):
        num_frames = duration
    else:
        raise AssertionError()
    return num_frames


@beartype
def torchaudio_load(
    data_source: Any,  # Union[str, BytesIO, ExFileObject] might also accept "tempfile.SpooledTemporaryFile" and many more
    offset: Optional[Union[int, float]] = None,
    duration: Optional[Union[int, float]] = None,
    format: Optional[str] = None,
    sample_rate: Optional[int] = None,
) -> tuple[TorchTensor1D, int]:
    if isinstance(data_source, BytesIO):
        assert format is not None, f"when reading from BytesIO a format must be given"
    signal, sample_rate = torchaudio.load(
        data_source,
        format=format,
        frame_offset=_parse_offset_for_torchaudio_load(offset, sample_rate),
        num_frames=_parse_duration_for_torchaudio_load(duration, sample_rate),
    )

    signal = get_first_channel(signal)
    assert (
        len(signal) > 1000
    ), f"{data_source=} below 1k samples is not really a signal!"
    return signal, sample_rate


@beartype
def get_first_channel(
    signal,  # Union[NumpyArray, TorchTensorFloat]
):  # Union[NumpyFloat1DArray, TorchTensor1D]:
    if len(signal.shape) == 2:
        channel_dim = np.argmin(signal.shape)
        first_channel = 0
        if channel_dim == 0:
            signal = signal[first_channel, :]
        else:
            signal = signal[:, first_channel]
    elif len(signal.shape) == 1:
        pass
    else:
        raise NotImplementedError("how to handle 3dim arrays?")

    signal = signal.squeeze()
    return signal


torchaudio.utils.sox_utils.set_buffer_size(
    16000
)  # necessary for long audio-headers (mp3)


@beartype
def load_resample_with_torch(
    data_source: Any,  # ExFileObject
    format: Optional[str] = None,
    sample_rate: Optional[int] = None,
    target_sample_rate: Optional[int] = 16000,
    offset: Optional[Union[int, float]] = None,
    duration: Optional[Union[int, float]] = None,
) -> TorchTensor1D:
    """
    Providing num_frames and frame_offset arguments will slice the resulting Tensor object while decoding.
    :param duration: see: .../torchaudio/backend/sox_io_backend.py
    """
    signal, sample_rate = torchaudio_load(
        data_source, offset, duration, format, sample_rate
    )
    signal = torchaudio_resample(signal.squeeze(), sample_rate, target_sample_rate)

    return signal

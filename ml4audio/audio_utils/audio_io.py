import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, Iterator, Any

import numpy as np
from beartype import beartype

from ml4audio.audio_utils.audio_data_models import FileLikeAudioDatum
from ml4audio.audio_utils.torchaudio_utils import load_resample_with_torch
from misc_utils.beartypes import NumpyFloat1DArray, NumpyInt16Dim1, Numpy1DArray
from misc_utils.processing_utils import exec_command
from misc_utils.utils import get_val_from_nested_dict, NOT_EXISTING

MAX_16_BIT_PCM: float = float(2 ** 15)  # 32768.0 for 16 bit, see "format"


@beartype
def load_audio_array_from_filelike(
    filelike: FileLikeAudioDatum,
    target_sample_rate: int,
    sample_rate: Optional[int] = None,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> NumpyFloat1DArray:
    # TODO: WTF why is fisher faster with nemo, but kaldi which is also wav, faster with torchaudio??
    if filelike.format == "wav" and not any(
        (s in filelike.audio_source for s in ["Fisher"])
    ):
        array = load_resample_with_nemo(
            audio_filepath=filelike.audio_source,
            offset=offset,
            duration=duration,
            target_sample_rate=target_sample_rate,
        )

    elif filelike.format in ["flac"]:
        # TODO: torchaudio cannot load flacs?
        array = load_resample_with_nemo(
            audio_filepath=BytesIO(filelike.audio_source.read()),
            offset=offset,
            duration=duration,
            target_sample_rate=target_sample_rate,
        )
    else:
        torch_tensor = load_resample_with_torch(
            data_source=filelike.audio_source,
            sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
            offset=offset if offset > 0.0 else None,
            duration=duration,
            format=filelike.format,
        )
        array = torch_tensor.numpy()

    return array


def ffprobe(vf):
    cmd = f'ffprobe -v error -print_format json -show_entries stream "{vf}"'
    o, e = exec_command(cmd)
    return json.loads("".join([x.decode("utf8") for x in o]))


def get_video_file_ffprobe_stream_infos(vf: str) -> list[dict]:
    info_data = ffprobe(vf)["streams"]

    def generate_infos():
        for s in info_data:
            kvs = (
                ("-".join(p), get_val_from_nested_dict(s, p))
                for p in [
                    ["codec_name"],
                    ["codec_type"],
                    ["tags", "language"],
                    ["tags", "title"],
                ]
            )
            yield {k: v for k, v in kvs if v != NOT_EXISTING}

    stream_infos = list(generate_infos())
    return stream_infos


def build_audio_cmd(af, k, vf):
    cmd = f'ffmpeg -i "{vf}" -y -filter_complex "[0:a:{k}]channelsplit=channel_layout=stereo[left][right]" -map "[left]" -c:a libopus -ar 16000 -ac 1 {af}_left.opus.ogg -map "[right]" -c:a libopus -ar 16000 -ac 1 {af}_right.opus.ogg'
    # cmd = f'ffmpeg -i "{vf}" -y -map 0:a:{k} -q:a 0 -ac 1 -ar 16000 "{af}.mp3"'
    return cmd


@beartype
def extract_streams_from_video_file(
    vf: str,
    audio_file_target_folder: str,
    stream_infos: Optional[list[dict]] = None,
    build_cmd_fun=build_audio_cmd,
    codec_type="audio",
) -> list[str]:
    if stream_infos is None:
        stream_infos = get_video_file_ffprobe_stream_infos(vf)

    audio_streams = [s for s in stream_infos if s["codec_type"] == codec_type]
    audio_files = []
    for k, info_d in enumerate(audio_streams):
        af = f"{audio_file_target_folder}/{vf}_lang_{info_d.get('tags-language', '')}_title_{info_d.get('tags-title', '')}_{k}"
        cmd = build_cmd_fun(af, k, vf)

        if not os.path.isfile(af):
            os.makedirs(Path(af).parent, exist_ok=True)
            print(exec_command(cmd))

        if not os.path.isfile(af):
            print(f"{af=} failed!")
        audio_files.append(af)

    return audio_files


@beartype
def load_and_resample_16bit_PCM(
    audio_filepath: str,
    target_sample_rate: int,
    offset=0.0,
    duration: Optional[float] = None,
) -> NumpyInt16Dim1:
    a = load_resample_with_nemo(audio_filepath, target_sample_rate, offset, duration)
    a = convert_to_16bit_array(a)
    # a = np.expand_dims(a, axis=1)  # TODO: why did I every want this?
    return a


@beartype
def convert_to_16bit_array(a: NumpyFloat1DArray) -> NumpyInt16Dim1:
    a = a / np.max(np.abs(a)) * (MAX_16_BIT_PCM - 1)
    a = a.astype(np.int16)
    return a


@beartype
def load_resample_with_nemo(
    audio_filepath: Union[str, BytesIO],
    target_sample_rate: Optional[int] = 16000,
    offset=0.0,
    duration: Optional[float] = None,
) -> NumpyFloat1DArray:
    from nemo.collections.asr.parts.preprocessing import AudioSegment

    # cause nemo wants 0 if no duration
    duration = 0 if duration is None else duration
    audio = AudioSegment.from_file(
        audio_filepath,
        target_sr=target_sample_rate,
        offset=offset,
        duration=duration,
        trim=False,
    )
    s = audio.samples.shape
    a = audio.samples.squeeze()
    return a


@beartype
def break_array_into_chunks(
    array: Numpy1DArray, chunk_size: int
) -> Iterator[Numpy1DArray]:
    """
    non-overlapping chunks
    """
    buffer = array.copy()
    while len(buffer) > 0:
        out = buffer[:chunk_size]
        buffer = buffer[chunk_size:]
        yield out


@beartype
def read_audio_chunks_from_file(
    audio_filepath: str,
    target_sample_rate: int,
    offset=0.0,
    duration=None,
    chunk_duration=0.05,
) -> Iterator[NumpyInt16Dim1]:
    """
    formerly named resample_stream_file
    """
    array = load_and_resample_16bit_PCM(
        audio_filepath, target_sample_rate, offset, duration
    )
    return break_array_into_chunks(array, int(target_sample_rate * chunk_duration))


@beartype
def normalize_audio_array(array: Numpy1DArray) -> NumpyFloat1DArray:
    """
    copypasted from nvidia/nemo: TranscodePerturbation
    """
    att_factor = 0.8  # to avoid saturation while writing to wav
    array = array.astype(np.float)

    max_level = np.max(np.abs(array))
    if max_level > 0.8:
        norm_factor = att_factor / max_level
        norm_samples = norm_factor * array
    else:
        norm_samples = array
    return norm_samples

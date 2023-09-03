import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Union, Iterator, Any, Annotated, Iterable

import ffmpeg
import librosa
import numpy as np
from beartype import beartype
from beartype.vale import Is
from numpy.typing import NDArray

from misc_utils.dataclass_utils import UNDEFINED
from ml4audio.audio_utils.audio_data_models import FileLikeAudioDatum
from ml4audio.audio_utils.overlap_array_chunker import (
    MessageChunk,
    messages_from_chunks,
)
from ml4audio.audio_utils.torchaudio_utils import (
    load_resample_with_torch,
    get_first_channel,
)
from misc_utils.beartypes import (
    NeNpFloatDim1,
    NumpyInt16Dim1,
    NpNumberDim1,
    TorchTensor1D,
    NpFloatDim1,
    File,
    NeNpFloatDim1,
    NpFloatDim1,
)
from misc_utils.processing_utils import exec_command
from misc_utils.utils import get_val_from_nested_dict, NOT_EXISTING
import soundfile as sf

MAX_16_BIT_PCM: float = float(2 ** 15)  # 32768.0 for 16 bit, see "format"


@beartype
def load_audio_array_from_filelike(
    filelike: FileLikeAudioDatum,
    target_sample_rate: int,
    sample_rate: Optional[int] = None,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> NeNpFloatDim1:
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
        #   nemo cannot properly handle multi-channel flacs
        audio_source = (
            filelike.audio_source
            if isinstance(filelike.audio_source, str)
            else BytesIO(filelike.audio_source.read())
        )
        array = load_resample_with_soundfile(
            audio_file=audio_source,
            target_sr=target_sample_rate,
            offset=offset,
            duration=duration,
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

    if len(array) < 1000:
        print(
            f"{filelike.audio_source=},{offset=},{duration=} below 1k samples is not really a signal!"
        )
    return array


def ffprobe(vf) -> dict:
    cmd = f'ffprobe -v error -print_format json -show_entries stream "{vf}"'
    o, e = exec_command(cmd)
    return json.loads("".join([x.decode("utf8") for x in o]))


@beartype
def ffprobe_audio_duration(vf: File) -> float:
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{vf}"'
    o, e = exec_command(cmd)
    return float(o[0])


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


# @beartype
# def load_and_resample_16bit_PCM(
#     audio_filepath: str,
#     target_sample_rate: int,
#     offset=0.0,
#     duration: Optional[float] = None,
# ) -> NumpyInt16Dim1:
#     a = load_resample_with_nemo(audio_filepath, target_sample_rate, offset, duration)
#     a = convert_to_16bit_array(a)
#     # a = np.expand_dims(a, axis=1)  # TODO: why did I every want this?
#     return a


@beartype
def convert_to_16bit_array(a: NpFloatDim1) -> NumpyInt16Dim1:
    a = a / np.max(np.abs(a)) * (MAX_16_BIT_PCM - 1)
    a = a.astype(np.int16)
    return a


def _convert_samples_to_float32(samples: NDArray) -> NDArray:
    """
    stolen from nemo
    Convert sample type to float32.
    Audio sample type is usually integer or float-point.
    Integers will be scaled to [-1, 1] in float32.
    """
    float32_samples = samples.astype("float32")
    if samples.dtype in np.sctypes["int"]:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= 1.0 / 2 ** (bits - 1)
    elif samples.dtype in np.sctypes["float"]:
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples


@beartype
def load_resample_with_soundfile(
    audio_file: Union[str, BytesIO],
    target_sr: Optional[int] = None,
    int_values: bool = False,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
    trim: bool = False,
    trim_db=60,
) -> NeNpFloatDim1:
    """
    based on nemo code: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/nemo/collections/asr/parts/preprocessing/segment.py#L173
    """
    with sf.SoundFile(audio_file, "r") as f:
        dtype = "int32" if int_values else "float32"
        sample_rate = f.samplerate
        if offset is not None:
            f.seek(int(offset * sample_rate))
        if duration is not None:
            samples = f.read(int(duration * sample_rate), dtype=dtype)
        else:
            samples = f.read(dtype=dtype)

    samples = (
        samples.transpose()
    )  # channels in first, signal in second axis, thats how librosa wants it

    samples = _convert_samples_to_float32(samples)
    if target_sr is not None and target_sr != sample_rate:
        samples = librosa.core.resample(
            samples, orig_sr=sample_rate, target_sr=target_sr
        )
    if trim:
        samples, _ = librosa.effects.trim(samples, top_db=trim_db)
    if samples.ndim >= 2:
        # here was bug in nemo-code!
        # explanation: resampy does resample very last axis, see: https://github.com/bmcfee/resampy/blob/29d34876a61fcd74e72003ceb0775abaf6fdb961/resampy/core.py#L15
        # resample(x, sr_orig, sr_new, axis=-1, filter='kaiser_best', **kwargs):
        assert samples.shape[0] < samples.shape[1]
        samples = np.mean(samples, 0)
    return samples


@beartype
def load_resample_with_nemo(
    audio_filepath: Union[str, BytesIO],
    target_sample_rate: Optional[int] = 16000,
    offset=0.0,
    duration: Optional[float] = None,
) -> NeNpFloatDim1:
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
    signal = audio.samples
    signal = get_first_channel(signal)
    assert (
        len(signal) > 1000
    ), f"{audio_filepath=} below 1k samples is not really a signal!"
    assert len(audio.samples.shape) == 1, f"{len(audio.samples.shape)=}"
    return signal


@beartype
def break_array_into_chunks(array: NDArray, chunk_size: int) -> Iterator[NDArray]:
    """
    non-overlapping chunks
    """
    buffer = array.copy()
    while len(buffer) > 0:
        out = buffer[:chunk_size]
        buffer = buffer[chunk_size:]
        yield out


@dataclass
class VarsizeNonOverlapChunker:
    array: NpNumberDim1

    def get_chunk(self, chunk_size: int) -> Optional[NpNumberDim1]:
        if len(self.array) > 0:
            out = self.array[:chunk_size]
            self.array = self.array[chunk_size:]
        else:
            out = None
        return out


@beartype
def read_audio_chunks_from_file(
    audio_filepath: str,
    target_sample_rate: int,
    offset=0.0,
    duration=None,
    chunk_duration=0.05,
) -> Iterator[NeNpFloatDim1]:
    """
    formerly named resample_stream_file
    """
    array = load_resample_with_torch(
        data_source=audio_filepath,
        target_sample_rate=target_sample_rate,
        offset=offset,
        duration=duration,
    )
    return break_array_into_chunks(
        array.numpy(), int(target_sample_rate * chunk_duration)
    )


@beartype
def normalize_audio_array(array: NpNumberDim1) -> NeNpFloatDim1:
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


Seconds = float


@beartype
def ffmpeg_load_audio_from_bytes(
    audio_bytes: bytes,
    sr: int = 16_000,
) -> NpFloatDim1:
    """
    based on: https://github.com/openai/whisper/blob/d18e9ea5dd2ca57c697e8e55f9e654f06ede25d0/whisper/audio.py#L22
    """
    try:
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True, input=audio_bytes)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _trimmed_input(
    inpt,
    start: Optional[Seconds] = None,
    end: Optional[Seconds] = None,
):
    """
    TODO(tilo): shit this is NOT working!
            I have the feeling of rewriting the wheel! this start/stop trimming should already be implemented in someones lib
    """
    if start and end:
        trimmed_input = inpt.trim(start=start, end=end)
    elif start:
        trimmed_input = inpt.trim(start=start)
    elif end:
        trimmed_input = inpt.trim(end=end)
    else:
        trimmed_input = inpt
    return trimmed_input


@beartype
def ffmpeg_load_audio_from_file(
    audio_file: File,
    sr: int = 16_000,
    # start: Optional[Seconds] = None,
    # end: Optional[Seconds] = None,
) -> NpFloatDim1:
    """
    based on: https://github.com/openai/whisper/blob/d18e9ea5dd2ca57c697e8e55f9e654f06ede25d0/whisper/audio.py#L22
    """
    try:
        # TODO: vf seems not to be working!
        # "ffmpeg -i audiomonolith/tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.wav -vf trim=1.11:2.22 -f s16le -ac 1 -acodec pcm_s16le -ar 1600 test.wav"
        cmd = ffmpeg.input(audio_file, threads=0).output(
            "-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr
        )
        out, _ = cmd.run(
            cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Failed to load audio: {e.stderr.decode()},{' '.join(cmd.get_args())=}"
        ) from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@beartype
def ffmpeg_load_trim(
    audio_file: File,
    sr: int = 16_000,
    start: Optional[Seconds] = None,
    end: Optional[Seconds] = None,
) -> NpFloatDim1:
    array = ffmpeg_load_audio_from_file(audio_file, sr)
    if start and end:
        array = array[round(start * sr) : round(end * sr)]
    elif start:
        array = array[round(start * sr) :]
    elif end:
        array = array[: round(end * sr)]

    return array


@beartype
def ffmpeg_torch_load(
    file: Annotated[str, Is[lambda f: os.path.isfile(f)]],
    target_sample_rate: int = 16000,
) -> TorchTensor1D:
    """
    TODO: this is super ugly, why cant I load with librosa? which or another ffmpeg wrapper
    """
    # name = Path(file).stem
    with NamedTemporaryFile(
        # prefix=name.replace(" ", "_"),
        suffix=".wav",
        delete=True,
    ) as tmp_wav:

        cmd = f'ffmpeg -i "{file}" -ac 1 -ar {target_sample_rate} {tmp_wav.name} -y'
        o, e = exec_command(cmd)

        audio = load_resample_with_torch(
            data_source=tmp_wav.name,
            target_sample_rate=target_sample_rate,
        )
    return audio


@dataclass
class AudioMessageChunk(MessageChunk):
    """
    instance of this represents one chunks of an audio-message
    an audio-message can be split into possibly overlapping chunks, entire message got one message_id
    frame_idx is counter/absolut-position of audio-chunk's start frame in entire audio-message
    """

    # message_id: str  # same for all chunks of same message
    # frame_idx: int  # points to very first frame of this chunk
    # array: Union[_UNDEFINED,NumpyInt16Dim1]=UNDEFINED
    array: NpFloatDim1 = (
        UNDEFINED  # TODO: former NumpyInt16Dim1 here, why was it like this?
    )
    chunk_idx: Optional[int] = None  # index of this chunk, TODO: who wanted this?


@beartype
def audio_messages_from_file(
    audio_filepath: str, client_sample_rate: int, chunk_duration: float = 0.1
) -> Iterator[AudioMessageChunk]:
    chunks = list(
        read_audio_chunks_from_file(
            audio_filepath, client_sample_rate, chunk_duration=chunk_duration
        )
    )
    yield from audio_messages_from_chunks(audio_filepath, chunks)


@beartype
def audio_messages_from_chunks(
    signal_id: str, chunks: Iterable[NpFloatDim1]
) -> Iterator[AudioMessageChunk]:
    for chunk_idx, m in enumerate(messages_from_chunks(signal_id, chunks)):
        yield AudioMessageChunk(
            message_id=m.message_id,
            frame_idx=m.frame_idx,
            array=m.array,
            chunk_idx=chunk_idx,
            end_of_signal=m.end_of_signal,
        )

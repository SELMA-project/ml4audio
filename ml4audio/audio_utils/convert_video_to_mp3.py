from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

from tqdm import tqdm

from ml4audio.audio_utils.audio_io import extract_streams_from_video_file
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.prefix_suffix import PrefixSuffix


@dataclass
class ExtractedMp3s(CachedData, Iterable[str]):
    name: Union[_UNDEFINED, str] = UNDEFINED
    path: Union[_UNDEFINED, str] = UNDEFINED
    audio_files: list[str] = field(init=False, default_factory=lambda: [])
    cache_base: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("processed_data", "extracted_mp3s")
    )

    def build_audio_cmd(self, af, k, vf):
        # cmd = f'ffmpeg -i "{vf}" -y -filter_complex "[0:a:{k}]channelsplit=channel_layout=stereo[left][right]" -map "[left]" -c:a libopus -ar 16000 -ac 1 {af}_left.opus.ogg -map "[right]" -c:a libopus -ar 16000 -ac 1 {af}_right.opus.ogg'
        cmd = f'ffmpeg -i "{vf}" -y -map 0:a:{k} -q:a 0 -ac 1 -ar 16000 "{af}.mp3"'
        return cmd

    def _build_cache(self):
        for p in tqdm(Path(self.path).rglob("*.mp4")):
            audio_files = extract_streams_from_video_file(
                str(p),
                audio_file_target_folder=self.prefix_cache_dir("mp3s"),
                build_cmd_fun=self.build_audio_cmd,
            )
            self.audio_files.extend(audio_files)

    def __iter__(self):
        for f in self.audio_files:
            yield f"{f}.mp3"

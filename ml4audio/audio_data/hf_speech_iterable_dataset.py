import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from beartype import beartype

from data_io.readwrite_files import (
    read_lines,
)
from ml4audio.audio_data.targz_asr_dataset import TarGzASRCorpus, TarGzTranscripts

HF_DATASETS = "huggingface_cache/datasets"


from nemo.utils import logging

logging.disabled = True
logging._logger = None  # TODO: WTF!! nemo is logging warnings at error level!!


@dataclass
class HFTarGzTranscripts(TarGzTranscripts):
    def contains_transcript(self, member: tarfile.TarInfo) -> bool:
        file_name = Path(member.name).stem
        return file_name in self.split_names

    @beartype
    def build_id_transcripts(
        self, split_name: str, transcript_files: list[str]
    ) -> list[Tuple[str, str]]:
        tsv_file = next(
            filter(lambda s: s.endswith(f"{split_name}.tsv"), transcript_files)
        )
        lines_g = read_lines(tsv_file)
        header = next(lines_g).split("\t")
        data = [{k: v for k, v in zip(header, l.split("\t"))} for l in lines_g]
        return [(d["path"], d["sentence"]) for d in data]

    def build_transcript_file_name(self, member_name: str) -> str:
        s = member_name.split("huggingface/datasets/downloads/extracted")[-1]
        return s.replace("/", "__")


@dataclass
class HFIterableDataset(TarGzASRCorpus):
    def is_audiofile(self, member_name: str) -> bool:
        return member_name.endswith(".mp3")

    def audiofile_to_id(self, member_name: str) -> str:
        return Path(member_name).name

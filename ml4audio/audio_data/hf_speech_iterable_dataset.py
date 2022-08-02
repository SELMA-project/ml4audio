import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from beartype import beartype

from data_io.readwrite_files import (
    read_lines,
)
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import UNDEFINED
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.audio_data.targz_asr_dataset import (
    TarGzASRCorpus,
    TarGzTranscripts,
    TarGzArrayTextWithSize,
)

HF_DATASETS = "huggingface_cache/datasets"


from nemo.utils import logging

logging.disabled = True
logging._logger = None  # TODO: WTF!! nemo is logging warnings at error level!!


@dataclass
class HFTarGzTranscripts(TarGzTranscripts):
    """
    TODO: rename to CommonVoiceTarGzTranscripts
    """

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


if __name__ == "__main__":

    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")

    corpus = TarGzArrayTextWithSize(
        corpus=HFIterableDataset(
            targztranscripts=HFTarGzTranscripts(
                targz_file=str(
                    PrefixSuffix(
                        "base_path",
                        "/data/ASR_DATA/COMMON_VOICE/cv-corpus-10.0-2022-07-04-es.tar.gz",
                    )
                ),
            ),
            split="dev",
        ),
        sample_rate=16000
        # limit=10
    ).build()

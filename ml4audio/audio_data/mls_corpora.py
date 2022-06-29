import tarfile
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype

from data_io.readwrite_files import read_lines
from misc_utils.beartypes import NeStr
from ml4audio.audio_data.targz_asr_dataset import TarGzASRCorpus, TarGzTranscripts


@dataclass
class MLSTarGzTranscripts(TarGzTranscripts):
    def contains_transcript(self, member: tarfile.TarInfo) -> bool:
        return member.name.endswith("transcripts.txt")

    @beartype
    def build_id_transcripts(
        self, split_name: str, transcript_files: list[str]
    ) -> list[tuple[str, NeStr]]: # NeStr too strict?
        t_file = next(
            filter(
                lambda s: s.endswith(f"{split_name}/transcripts.txt"), transcript_files
            )
        )

        @beartype
        def parse_line(l: str) -> tuple[str, NeStr]:
            eid, transcript = l.split("\t")
            return eid, transcript

        return [parse_line(l) for l in read_lines(t_file)]

    def build_transcript_file_name(self, member_name: str) -> str:
        return member_name.split("huggingface/datasets/downloads/extracted")[-1]


class MLSIterableDataset(TarGzASRCorpus):
    def audiofile_to_id(self, member_name: str) -> str:
        return Path(member_name).name.replace(".flac", "")

    def is_audiofile(self, member_name: str) -> bool:
        return member_name.endswith(".flac")

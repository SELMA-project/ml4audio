import os
import shutil
import subprocess
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Annotated, Any, Optional

from beartype import beartype
from beartype.vale import Is
from tqdm import tqdm

from data_io.readwrite_files import read_lines, write_lines
from misc_utils.beartypes import NeList
from misc_utils.buildable_data import BuildableData
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from misc_utils.processing_utils import exec_command
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer
from ml4audio.text_processing.kenlm_arpa import ArpaBuilder, GotArpaFile, ArpaFile

Model_Unigrams_File = tuple[str, str]


@beartype
def build_unigrams_from_arpa(
    arpa_file: ArpaFile, transcript_normalizer: TranscriptNormalizer
) -> NeList[str]:
    def gen_parse_arpa_file():
        for line in read_lines(arpa_file):
            if "2-grams:" in line:
                break
            elif len(line.split("\t")) < 2:
                continue
            else:
                yield line.split("\t")[1]

    unigrams = list(
        {
            l
            for raw in tqdm(
                gen_parse_arpa_file(),
                desc="building unigrams, the LMs vocabulary",
            )
            for l in transcript_normalizer.apply(raw).split(" ")
        }
    )

    if len(unigrams) < 10_000:
        print(f"only got {len(unigrams)} unigrams!")
    assert all(" " not in s for s in unigrams)
    return unigrams


@dataclass
class NgramLmAndUnigrams(BuildableData):
    base_dir: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["lm_models"])

    @property
    @abstractmethod
    def ngramlm_filepath(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def unigrams_filepath(self) -> Optional[str]:
        """
        pyctcdecode can also decode without a unigrams-file, even though it throws some warnings
        """
        raise NotImplementedError

    @property
    def _is_data_valid(self) -> bool:
        no_need = not self.unigrams_filepath
        need_and_got_unigrams = no_need or os.path.isfile(self.unigrams_filepath)
        return os.path.isfile(self.ngramlm_filepath) and need_and_got_unigrams


@dataclass
class GzippedArpaAndUnigramsForPyCTCDecode(NgramLmAndUnigrams):
    raw_arpa: GotArpaFile = UNDEFINED
    transcript_normalizer: Union[_UNDEFINED, TranscriptNormalizer] = UNDEFINED

    @property
    def name(self):
        return f"gzipped_arpa_unigrams-{self.raw_arpa.name}"

    @property
    def ngramlm_filepath(self) -> str:
        return f"{self.data_dir}/lm.arpa.gz"

    @property
    def unigrams_filepath(self) -> str:
        return f"{self.data_dir}/unigrams.txt.gz"

    def _build_data(self) -> Any:
        self._gzip_or_copy_arpa()
        unigrams = build_unigrams_from_arpa(
            self.ngramlm_filepath, transcript_normalizer=self.transcript_normalizer
        )
        write_lines(self.unigrams_filepath, unigrams)

    def _gzip_or_copy_arpa(self):
        raw_arpa_file = self.raw_arpa.arpa_filepath
        assert os.path.isfile(raw_arpa_file), f"could not find {self.raw_arpa=}"
        if not raw_arpa_file.endswith(".gz"):
            out_err = exec_command(f"gzip -c {raw_arpa_file} > {self.ngramlm_filepath}")
            print(f"{out_err=}")
        else:
            shutil.copy(raw_arpa_file, self.ngramlm_filepath)
        assert os.path.isfile(self.ngramlm_filepath)


@dataclass
class KenLMBinaryUnigramsFile(NgramLmAndUnigrams):

    name: str = UNDEFINED
    kenlm_binary_file: PrefixSuffix = UNDEFINED
    unigrams_file: Optional[PrefixSuffix] = None

    @property
    def ngramlm_filepath(self) -> str:
        return f"{self.data_dir}/{Path(str(self.kenlm_binary_file)).name}"

    @property
    def unigrams_filepath(self) -> Optional[str]:
        return (
            f"{self.data_dir}/{Path(str(self.unigrams_file)).name}"
            if self.unigrams_file is not None
            else None
        )

    def _build_data(self) -> Any:
        shutil.copy(str(self.kenlm_binary_file), self.ngramlm_filepath)
        if self.unigrams_filepath:
            shutil.copy(str(self.unigrams_file), self.unigrams_filepath)


def build_binary_kenlm(kenlm_bin_path: str, arpa_file: str, kenlm_binary_file: str):
    """
    based on: "make_kenlm" method from https://github.com/NVIDIA/NeMo/blob/e859e43ef85cc6bcdde697f634bb3b16ee16bc6b/scripts/asr_language_modeling/ngram_lm/ngram_merge.py#L286
    Builds a language model from an ARPA format file using the KenLM toolkit.
    """
    sh_args = [
        os.path.join(kenlm_bin_path, "build_binary"),
        "trie",
        "-i",
        arpa_file,
        kenlm_binary_file,
    ]
    return subprocess.run(
        sh_args,
        capture_output=False,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


@dataclass
class KenLMBinaryUnigramsFromArpa(NgramLmAndUnigrams):

    arpa_unigrams: GzippedArpaAndUnigramsForPyCTCDecode = UNDEFINED

    @property
    def name(self):
        return f"binary-{self.arpa_unigrams.name}"

    @property
    def ngramlm_filepath(self) -> str:
        return f"{self.data_dir}/kenlm.bin"

    @property
    def unigrams_filepath(self) -> str:
        return f"{self.data_dir}/unigrams.txt.gz"

    def _build_data(self) -> Any:
        build_binary_kenlm(
            "/opt/kenlm/bin",
            self.arpa_unigrams.ngramlm_filepath,
            self.ngramlm_filepath,
        )
        shutil.copy(str(self.arpa_unigrams.unigrams_filepath), self.unigrams_filepath)

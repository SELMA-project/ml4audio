import os
import shutil
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Union, Annotated

from beartype import beartype
from beartype.vale import Is
from tqdm import tqdm

from data_io.readwrite_files import read_lines, write_lines
from misc_utils.beartypes import NeList
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from misc_utils.processing_utils import exec_command
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer
from ml4audio.text_processing.kenlm_arpa import ArpaBuilder

Model_Unigrams_File = tuple[str, str]

arpa_suffixes = [".arpa.gz", ".arpa", ".gz"]  # TODO: WTF! who calls a arpa "lm.gz"?
ArpaFile = Annotated[
    str,
    Is[lambda s: any(s.endswith(suffix) for suffix in arpa_suffixes)],
]


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
    assert all(" " not in s for s in unigrams)
    return unigrams


@dataclass
class KenLMForPyCTCDecode(CachedData):
    """
    TODO: rename to ArpaFile for pyctcdecode
    """

    transcript_normalizer: Union[_UNDEFINED, TranscriptNormalizer] = UNDEFINED
    cache_base: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["lm_models"])

    @property
    @abstractmethod
    def _raw_arpa_filepath(self) -> str:
        raise NotImplementedError

    @property
    def arpa_filepath(self) -> str:
        return self.prefix_cache_dir("lm.arpa.gz")

    @property
    def unigrams_filepath(self) -> str:
        return self.prefix_cache_dir("unigrams.txt.gz")

    def _build_cache(self):
        assert os.path.isfile(
            self._raw_arpa_filepath
        ), f"could not find {self._raw_arpa_filepath=}"
        if not self._raw_arpa_filepath.endswith(".gz"):
            out_err = exec_command(
                f"gzip -c {self._raw_arpa_filepath} > {self.arpa_filepath}"
            )
            print(f"{out_err=}")
        else:
            shutil.copy(self._raw_arpa_filepath, self.arpa_filepath)

        unigrams = build_unigrams_from_arpa(
            self.arpa_filepath, transcript_normalizer=self.transcript_normalizer
        )
        # KenLMBinaryAndLexicon() # TODO
        write_lines(self.unigrams_filepath, unigrams)


@dataclass
class KenLMForPyCTCDecodeFromArpaCorpus(KenLMForPyCTCDecode):
    arpa_builder: Union[ArpaBuilder, _UNDEFINED] = UNDEFINED

    @property
    def _is_ready(self) -> bool:
        is_ready = super()._is_ready
        if is_ready:
            # just to show-case an example of a build-time dependency
            assert (
                not self.arpa_builder._was_built
            ), f"arpa_builder is build-time dependency not loaded if KenLMForPyCTCDecodeFromArpa is loaded from cache!"
        return is_ready

    @property
    def name(self):
        return self.arpa_builder.name

    @property
    def _raw_arpa_filepath(self) -> str:
        return self.arpa_builder.arpa_filepath


@dataclass
class KenLMForPyCTCDecodeFromArpa(KenLMForPyCTCDecode):
    name: Union[_UNDEFINED, str] = UNDEFINED
    arpa_file: Union[ArpaFile, _UNDEFINED] = UNDEFINED

    @property
    def _raw_arpa_filepath(self) -> str:
        return self.arpa_file

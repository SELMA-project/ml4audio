import os
import shutil
import subprocess
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Annotated

import sys
from beartype import beartype
from beartype.vale import Is

from data_io.readwrite_files import read_lines
from misc_utils.beartypes import NeStr
from misc_utils.cached_data import CachedData

# based on: https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py
from misc_utils.dataclass_utils import (
    UNDEFINED,
    _UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.text_processing.word_based_text_corpus import WordBasedLMCorpus

# TODO: move this to its own package? cause it depends on kenlm


@dataclass
class ArpaArgs:
    order: int = 3
    max_memory: str = "80%"
    prune: str = "0|8|9"
    kenlm_bin: str = "/opt/kenlm/bin"
    vocab_size: Optional[int] = None


arpa_suffixes = [".arpa.gz", ".arpa", ".gz"]  # TODO: WTF! who calls a arpa "lm.gz"?
ArpaFile = Annotated[
    str,
    Is[lambda s: any(s.endswith(suffix) for suffix in arpa_suffixes)],
]


class GotArpaFile:
    name: NeStr
    arpa_filepath: ArpaFile = field(init=False)


@dataclass
class AnArpaFile(GotArpaFile):
    arpa_filepath: ArpaFile = field(init=True)

    def __post_init__(self):
        self.name = Path(self.arpa_filepath).name


@dataclass
class ArpaBuilder(CachedData, GotArpaFile):
    arpa_args: Union[_UNDEFINED, ArpaArgs] = UNDEFINED
    corpus: Union[_UNDEFINED, WordBasedLMCorpus] = UNDEFINED
    cache_base: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["lm_models"])

    @property
    def name(self):
        return f"arpa-{self.corpus.name}"

    @property
    def arpa_filepath(self) -> str:
        return self.prefix_cache_dir("lm.arpa")

    def _build_cache(self):
        corpus_file, word_counts_file = (
            self.corpus.corpus_filepath,
            self.corpus.word_counts_filepath,
        )
        if word_counts_file is not None:
            vocab_str = "\n".join(
                l.split("\t")[0]
                for l in read_lines(word_counts_file, limit=self.arpa_args.vocab_size)
            )
        else:
            vocab_str = None

        build_kenlm_arpa(
            self.arpa_args,
            str(self.cache_dir),
            self.arpa_filepath,
            corpus_file,
            vocab_str,
        )
        assert os.path.isfile(self.arpa_filepath), f"could build {self.arpa_filepath=}"


@beartype
def build_kenlm_arpa(
    args: ArpaArgs,
    output_dir: str,
    arpa_file: str,
    text_file: str,
    vocab_str: Optional[str] = None,
):
    print("\nCreating ARPA file ...")
    os.makedirs(output_dir, exist_ok=True)
    subargs = [
        os.path.join(args.kenlm_bin, "lmplz"),
        "--order",
        str(args.order),
        "--temp_prefix",
        output_dir,
        "--memory",
        args.max_memory,
        "--text",
        text_file,
        "--arpa",
        arpa_file,
        "--prune",
        *args.prune.split("|"),
        "--skip_symbols",
        "--discount_fallback",
    ]
    subprocess.check_call(subargs, stdout=sys.stdout, stderr=sys.stdout)

    if vocab_str is not None:
        # Filter LM using vocabulary of top-k words
        print("\nFiltering ARPA file using vocabulary of top-k words ...")
        arpa_file_unfiltered = f"{output_dir}/lm_unfiltered.arpa"
        shutil.copy(arpa_file, arpa_file_unfiltered)

        subprocess.run(
            [
                os.path.join(args.kenlm_bin, "filter"),
                "single",
                f"model:{arpa_file_unfiltered}",
                arpa_file,
            ],
            input=vocab_str.encode("utf-8"),
            check=True,
        )

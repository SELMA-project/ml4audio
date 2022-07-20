import json
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

from tqdm import tqdm

from data_io.readwrite_files import read_lines, write_lines
from misc_utils.beartypes import NeList
from misc_utils.buildable import BuildableList
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import (
    UNDEFINED,
    FILLME,
    _UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from misc_utils.utils import get_val_from_nested_dict
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer


@dataclass
class RglobRawCorpus(CachedData):
    corpus_dir: str = "some-dir"
    file_pattern: str = "file-pattern"
    cache_base: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["lm_data"])
    upsample_factor: int = 1

    @property
    def name(self):
        return Path(self.corpus_dir).stem

    @property
    def corpus_filepath(self):
        return self.prefix_cache_dir("raw_corpus.txt.gz")

    def get_raw_text_fun(self, line: str):
        return line

    def _build_cache(self):
        counter = [0]

        def count_lines(l):
            counter[0] += 1  # TODO this is ugly!
            return l

        files = self._get_files()
        print(f"{self.name} found {len(files)} files: {files=}")
        lines_g = (
            count_lines(self.get_raw_text_fun(line))
            for file in files
            for line in read_lines(file)
        )
        write_lines(
            self.corpus_filepath, tqdm(lines_g, f"{self.name} is writing lines")
        )
        assert counter[0] > 0, f"{self.name} got zero lines!"

    def _get_files(self) -> NeList[str]:
        return list(str(f) for f in Path(self.corpus_dir).rglob(self.file_pattern))

    def __iter__(self):
        for _ in range(self.upsample_factor):
            yield from read_lines(self.corpus_filepath)


@dataclass
class RglobRawCorpusFromDicts(RglobRawCorpus):
    dict_path: str = "key_a.key_b.key_c"

    def get_raw_text_fun(self, line: str):
        return get_val_from_nested_dict(json.loads(line), self.dict_path.split("."))


def spacesplit_tokenize_and_tokencounting(line, counter):
    text = line.replace("\n", "").replace("\r", "")
    if len(text) > 9:
        tokens = text.split(" ")  # TODO: proper tokenization!? why space?
        counter.update(tokens)
    else:
        text = None

    return text


@dataclass
class WordBasedLMCorpus(CachedData):
    name: Union[_UNDEFINED, str] = field(init=True, default=UNDEFINED)
    raw_corpora: Union[_UNDEFINED, BuildableList[RglobRawCorpus]] = UNDEFINED
    normalizer: FILLME[TranscriptNormalizer] = UNDEFINED
    cache_base: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["lm_data"])

    @property
    def corpus_filepath(self) -> str:
        return self.prefix_cache_dir("processed.txt.gz")

    @property
    def word_counts_filepath(self) -> str:
        return self.prefix_cache_dir("word_counts.txt")

    def process_line(self, l: str, counter):
        s = self.normalizer.apply(l)
        return spacesplit_tokenize_and_tokencounting(s, counter)

    def _build_cache(self):

        lines_g = (line for corpus in self.raw_corpora.data for line in corpus)

        counter = Counter()

        write_lines(
            self.corpus_filepath,
            tqdm(
                filter(
                    lambda x: x is not None,
                    map(partial(self.process_line, counter=counter), lines_g),
                ),
                desc=f"{self.name} is writing processed text_file",
            ),
        )
        wordcounts: dict[str, int] = {
            word: count
            for word, count in sorted(counter.items(), key=lambda kv: -kv[1])
        }
        assert len(wordcounts) > 0, f"{self.name} contains no words!"
        write_lines(
            self.word_counts_filepath,
            (f"{word}\t{count}" for word, count in wordcounts.items()),
        )

# based on: https://github.com/NVIDIA/NeMo/blob/3d0c29a317b89b20c93757010db80271eeea6816/examples/nlp/token_classification/data/get_tatoeba_data.py
import itertools
import os
import re
import string
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Union, Optional

from beartype import beartype
from tqdm import tqdm

from data_io.readwrite_files import write_lines, read_lines
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from misc_utils.processing_utils import iterable_to_batches
from nemo_punctuation_capitalization.punctcap_training.punctuation_tatoeba_data import (
    TatoebaWikipediaData,
    TatoebaMonolingualData,
)


@beartype
def split_into_train_dev(
    lines: Iterable[str], train_file: str, dev_file: str, dev_size: int
) -> None:
    """
    # based on: "__split_into_train_dev":  https://github.com/NVIDIA/NeMo/blob/3d0c29a317b89b20c93757010db80271eeea6816/examples/nlp/token_classification/data/get_tatoeba_data.py
    tilo: not very sophisticated! simply puts first "dev_size"
    samples of iterator in dev_file and rest in train_file
    -> but thats what nemo proposed!

    """

    it = iter(lines)

    def resuse_iterator(iterator):
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break

    write_lines(
        dev_file, (next(it) for _ in tqdm(range(dev_size), desc=f"writing {dev_file}"))
    )
    write_lines(train_file, tqdm(resuse_iterator(it), desc=f"writing {train_file}"))


def remove_punctuation(word: str) -> str:
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation.replace("'", "")
    return re.sub("[" + all_punct_marks + "]", "", word)


def generate_token_labels(
    lines: Iterable[str], punct_marks: str
) -> Iterator[tuple[str, str]]:
    for line in tqdm(lines, desc="preparing text label files"):
        line = line.split()  # TODO: thats very simple!
        for o_word in line:
            label = o_word[-1] if o_word[-1] in punct_marks else "O"
            word = remove_punctuation(o_word)
            if len(word) > 0:
                if word[0].isupper():
                    label += "U"
                else:
                    label += "O"

                word = word.lower()
                yield o_word, word, label


@beartype
def create_text_and_labels(
    output_dir: str,
    lines: Iterable[str],
    file_suffix: str,
    max_seq_len=128,
    punct_marks: str = ",.?",
):
    """
    based on: # based on: "create_text_and_labels" from https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/data/get_tatoeba_data.py

    Create datasets for training and evaluation.

    Args:
      output_dir: path to the output data directory
      file_path: path to file name
      punct_marks: supported punctuation marks

    The data will be split into 2 files: text.txt and labels.txt. \
    Each line of the text.txt file contains text sequences, where words\
    are separated with spaces. The labels.txt file contains \
    corresponding labels for each word in text.txt, the labels are \
    separated with spaces. Each line of the files should follow the \
    format:  \
    [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
    [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
    """

    labels_file = os.path.join(output_dir, "labels_" + file_suffix)
    text_file = os.path.join(output_dir, "text_" + file_suffix)
    original_file = os.path.join(output_dir, "original_" + file_suffix)

    with open(text_file, mode="wb") as text_f, open(
        labels_file, "wb"
    ) as labels_f, open(original_file, "wb") as original_f:
        g = generate_token_labels(lines, punct_marks)
        batches_g = iterable_to_batches(g, batch_size=max_seq_len)
        for words_tokens_labels in tqdm(batches_g, desc="write text labels files"):
            o_words, tokens, labels = [list(x) for x in zip(*words_tokens_labels)]
            assert len(tokens) <= max_seq_len
            assert len(labels) <= max_seq_len
            assert len(labels) == len(tokens)
            s = " ".join(tokens).strip() + "\n"
            text_f.write(s.encode("utf-8"))
            s = " ".join(labels).strip() + "\n"
            labels_f.write(s.encode("utf-8"))
            s = " ".join(o_words).strip() + "\n"
            original_f.write(s.encode("utf-8"))


@dataclass
class NepucaSplit(CachedData):
    """
    Nemo Punctuation Capitalization == Nepuca
    """

    name: Union[_UNDEFINED, str] = UNDEFINED
    raw_lines: Union[_UNDEFINED, Iterable[str]] = UNDEFINED
    limit: Optional[int] = None
    dev_size: int = 10_000

    cache_base: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix(
            "processed_data", "PUNCTUATION_CAPITALIZATION"
        )
    )

    @property
    def train_file(self):
        return self.prefix_cache_dir(f"train.txt.gz")

    @property
    def dev_file(self):
        return self.prefix_cache_dir(f"dev.txt.gz")

    def _build_cache(self):
        lines_g = itertools.islice(self.raw_lines, 0, self.limit)
        split_into_train_dev(
            lines_g, self.train_file, self.dev_file, dev_size=self.dev_size
        )


@dataclass
class NepucaData(CachedData):
    train_dev_data: Union[_UNDEFINED, NepucaSplit] = UNDEFINED
    max_seq_len: int = 64
    cache_base: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix(
            "processed_data", "PUNCTUATION_CAPITALIZATION"
        )
    )

    @property
    def data_dir(self):
        return self.prefix_cache_dir("data")

    @property
    def name(self):
        return self.train_dev_data.name

    def _build_cache(self):
        os.makedirs(self.data_dir, exist_ok=True)

        create_text_and_labels(
            self.data_dir,
            read_lines(self.train_dev_data.train_file),
            max_seq_len=self.max_seq_len,
            file_suffix="train.txt",
        )
        create_text_and_labels(
            self.data_dir,
            read_lines(self.train_dev_data.dev_file),
            max_seq_len=self.max_seq_len,
            file_suffix="dev.txt",
        )


if __name__ == "__main__":
    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")
    BASE_PATHES["processed_data"] = PrefixSuffix("cache_root", "PROCESSED_DATA")

    lang_code = "por"
    wikipedia_data = TatoebaWikipediaData(
        raw_data=TatoebaMonolingualData(
            base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
            file_name=f"{lang_code}.tar",
        )
    )
    NepucaData(
        train_dev_data=NepucaSplit(
            name=wikipedia_data.name, raw_lines=wikipedia_data, limit=10_000
        ),
    ).build()

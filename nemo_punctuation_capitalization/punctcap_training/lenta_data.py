import itertools
import os.path
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import pandas
import regex
from pandas import Series
from tqdm import tqdm

from data_io.download_extract_files import wget_file
from data_io.readwrite_files import write_lines
from misc_utils.buildable import Buildable
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES


def clean_russian_text_for_punctcap_training(text: str, punct_marks: str = ",?."):
    """
    copypasted from tugtekins "russian_normalization_train"-method <fhg-gitlab>/mturan/russian-punct-casing/-/blob/master/local/get_lenta_data.py#L13
    # TODO(tilo): @tugtekin -> please explanation for: NFKC, re.sub, regex.sub, !
        we need pytests!
        written2spoken before this cleaning method, OR adapt cleaning method as to keep numbers!
    Normalization and deRomanization of Russian data for training purpose
    Args:
        text: input text
        punct_marks: supported punctuation marks
    """
    # \xa0 (i.e. chr(160)) creates problems in Lenta dataset
    # this is a non-breaking space in Latin1 (ISO 8859-1)
    # for other languages (with Latin alphabet) better to use 'NFKD' instead of 'NFKC'
    unicoded = unicodedata.normalize("NFKC", text)

    # remove links if exist
    no_URL = re.sub(r"http\S+", "", unicoded)

    # delete alphanumeric Latin letters except for Cyrillic (no transliteration)
    # also, remove all the punctuations except defined in 'punct_marks'
    match = "[^\s\p{IsCyrillic}" + punct_marks + "]"
    only_cyrillic_and_punctuations = regex.sub(match, "", no_URL)

    # remove repetitive whitespace
    normalized = " ".join(only_cyrillic_and_punctuations.split())

    # replace punctuations with extra spaces
    # e.g. "hey . you are , okay ?" --> "hey. you are, okay?"
    remove_extra_space = {" ,": ",", " .": ".", " ?": "?"}
    for key, value in remove_extra_space.items():
        normalized = normalized.replace(key, value)

    return normalized


@dataclass
class LentaData(Buildable, Iterable[str]):

    punct_marks: str = ",?."  # TODO: !

    data_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("raw_data", "russian-lenta-data")
    )

    _bz2_file_url: str = field(
        init=False,
        default="https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2",
    )

    @property
    def raw_file(self):
        return f'{self.data_dir}/{self._bz2_file_url.split("/")[-1]}'

    def _build_self(self) -> "LentaData":
        if not os.path.isfile(self.raw_file):
            wget_file(self._bz2_file_url, str(self.data_dir))
        return self

    def __iter__(self)->Iterator[str]:
        chunksize = 1000

        with pandas.read_csv(
            self.raw_file, usecols=["text"], index_col=False, chunksize=chunksize
        ) as reader:
            for chunk in reader:
                for d in chunk.iterrows():
                    if isinstance(d,tuple) and isinstance(d[1],Series) and isinstance(d[1].text,str):
                        original = d[1].text.replace("\n", " ").replace("\r", "")
                        yield original
                    # clean_russian_text_for_punctcap_training(
                    #     original, punct_marks=self.punct_marks
                    # )


if __name__ == "__main__":

    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")
    BASE_PATHES["processed_data"] = PrefixSuffix("cache_root", "PROCESSED_DATA")

    corpus = LentaData().build()
    list(tqdm(corpus))
    for t in itertools.islice(corpus, 0, 10):
        print(f"{t=}")
    # wget -q ${lenta} -P ${resource_fold} || exit 1
    # bzip2 -d ${resource_fold}/${file_base} || exit 1

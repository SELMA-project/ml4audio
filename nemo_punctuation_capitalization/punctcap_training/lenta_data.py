import itertools
import os.path
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Any

import pandas
from data_io.download_extract_files import wget_file
from misc_utils.buildable_data import BuildableData
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from pandas import Series

"""
this code is completely unused!
def clean_russian_text_for_punctcap_training(text: str, punct_marks: str = ",?."):
    copypasted from tugtekins "russian_normalization_train"-method <fhg-gitlab>/mturan/russian-punct-casing/-/blob/master/local/get_lenta_data.py#L13
    # TODO(tilo): @tugtekin -> please explanation for: NFKC, re.sub, regex.sub, !
        we need pytests!
        written2spoken before this cleaning method, OR adapt cleaning method as to keep numbers!
    Normalization and deRomanization of Russian data for training purpose
    Args:
        text: input text
        punct_marks: supported punctuation marks
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
"""


def got_text(d) -> bool:
    return (
        isinstance(d, tuple) and isinstance(d[1], Series) and isinstance(d[1].text, str)
    )


@dataclass
class LentaData(BuildableData, Iterable[str]):

    """
    is simply downloading the lenta-ru-news.csv.bz2 file into the "russian_text_data"-folder
    """

    punct_marks: str = ",?."  # TODO: !

    base_dir: PrefixSuffix = field(
        default_factory=lambda: BASE_PATHES["russian_text_data"]
    )

    _bz2_file_url: str = field(
        init=False,
        default="https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2",
    )

    @property
    def name(self) -> str:
        return "russian-lenta-data"

    @property
    def _is_data_valid(self) -> bool:
        return os.path.isfile(self.raw_file)

    @property
    def raw_file(self):
        return f'{self.data_dir}/{self._bz2_file_url.split("/")[-1]}'

    def _build_data(self) -> Any:
        os.makedirs(str(self.data_dir), exist_ok=True)
        if not os.path.isfile(self.raw_file):
            wget_file(self._bz2_file_url, str(self.data_dir))
        return self

    def __iter__(self) -> Iterator[str]:
        chunksize = 1000

        with pandas.read_csv(
            self.raw_file, usecols=["text"], index_col=False, chunksize=chunksize
        ) as reader:
            rows = (chunk for chunk in reader for d in chunk.iterrows())
            good_rows = (d for d in rows if got_text(d))
            for d in good_rows:
                original = d[1].text.replace("\n", " ").replace("\r", "")
                yield original


if __name__ == "__main__":

    processed_corproa_dir = f"{os.environ['corpora']}/processed_corpora"
    BASE_PATHES["processed_corproa_dir"] = processed_corproa_dir
    BASE_PATHES["russian_text_data"] = PrefixSuffix(
        "processed_corproa_dir", "RUSSIAN_TEXT_DATA"
    )

    corpus = LentaData().build()

    for t in itertools.islice(corpus, 0, 10):
        print(f"{t=}")

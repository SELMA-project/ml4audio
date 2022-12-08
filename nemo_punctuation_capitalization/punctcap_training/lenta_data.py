import itertools
import os.path
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Any

import pandas
import regex
from misc_utils.buildable_data import BuildableData
from pandas import Series
from tqdm import tqdm

from data_io.download_extract_files import wget_file
from data_io.readwrite_files import write_lines
from misc_utils.buildable import Buildable
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES


@dataclass
class LentaData(BuildableData, Iterable[str]):

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

    def got_text(d) -> bool:
        return (
            isinstance(d, tuple)
            and isinstance(d[1], Series)
            and isinstance(d[1].text, str)
        )


if __name__ == "__main__":

    base_path = os.environ["BASE_PATH"]
    processed_corproa_dir = "/nm-raid/audio/data/corpora/processed_corpora"
    # cache_root = f"{base_path}/data/cache"
    BASE_PATHES["processed_corproa_dir"] = processed_corproa_dir
    # BASE_PATHES["processed_data"] = PrefixSuffix("cache_root", "PROCESSED_DATA")

    corpus = LentaData().build()
    list(tqdm(corpus))
    for t in itertools.islice(corpus, 0, 10):
        print(f"{t=}")
    # wget -q ${lenta} -P ${resource_fold} || exit 1
    # bzip2 -d ${resource_fold}/${file_base} || exit 1

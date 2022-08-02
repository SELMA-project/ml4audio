import os

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, ClassVar, Iterable, Iterator, Any, Optional

from data_io.download_extract_files import download_data, wget_file
from data_io.readwrite_files import read_lines
from misc_utils.buildable import Buildable
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import UNDEFINED, _UNDEFINED, encode_dataclass
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from tqdm import tqdm


@dataclass
class TatoebaMonolingualData(Buildable):
    base_url: Union[
        _UNDEFINED, str
    ] = UNDEFINED  # https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28/deu.tar
    file_name: Union[_UNDEFINED, str] = UNDEFINED
    data_base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("raw_data", "TATOEBA_WIKIPEDIA_DATA")
    )

    @property
    def name(self):
        return self.file_name.replace(".tar", "")

    def _build_self(self) -> Any:
        if not os.path.isdir(
            f"{self.data_dir}/data"
        ):  # for german it makes a deu/deu/data path
            print(f"downloading: {self.name}")
            download_data(
                base_url=self.base_url,
                file_name=self.file_name,
                data_dir=str(self.data_base_dir),
                unzip_it=True,
                remove_zipped=False,
                verbose=True,
            )
        else:
            print(f"found data in {self.data_dir}")

    @property
    def data_dir(self):
        return f"{self.data_base_dir}/{self.name}"


@dataclass
class TatoebaWikipediaData(Buildable, Iterable[str]):
    raw_data: Union[_UNDEFINED, TatoebaMonolingualData] = UNDEFINED
    num_lines: Optional[int] = None

    @property
    def name(self):
        return self.raw_data.name

    def _build_self(self) -> Any:
        self.num_lines = sum((1 for l in self))

    def __iter__(self):
        files = list(Path(self.raw_data.data_dir).rglob("wikipedia.txt.gz"))
        if len(files) > 0:
            file = files[0]
            yield from read_lines(str(file))
        else:
            print(f"{self.name} does not have wikipedia.txt.gz")

    # def __len__(self):
    #     return self.num_lines


@dataclass
class TatoebaLanguages(CachedData, Iterable[str]):
    url: ClassVar[
        str
    ] = "https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/master/data/MonolingualData.md"
    lang_codes: list[str] = field(init=False)
    cache_base: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("raw_data", "TATOEBA")
    )

    @property
    def name(self):
        return "tatoeba-monolingual-data"

    def _build_cache(self):
        wget_file(self.url, self.prefix_cache_dir("data"))
        self.lang_codes = [
            l.replace("* [", "").split("]")[0]
            for l in read_lines(f'{self.prefix_cache_dir("data")}/MonolingualData.md')
            if l.startswith("* [") and l.endswith(".tar)")
        ]

    def __iter__(self) -> Iterator[str]:
        yield from self.lang_codes


if __name__ == "__main__":
    """
    cd .../audiopolylith/text_processing
    # pyhton-path needed for proper module/target resolution cause here I run from main!
    export PYTHONPATH=${PWD}
    """
    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")

    # lang_codes = ListOfLanguages().build()
    # print(list(lang_codes))
    lang_codes = ["eng", "deu", "fra", "por", "spa", "ita"]
    for lang_code in tqdm(lang_codes, desc="languages"):
        TatoebaMonolingualData(
            base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
            file_name=f"{lang_code}.tar",
        ).build()

    lang_code = "deu"
    wikipedia_data = TatoebaWikipediaData(
        raw_data=TatoebaMonolingualData(
            base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
            file_name=f"{lang_code}.tar",
        )
    ).build()
    for l in itertools.islice(wikipedia_data, 0, 10):
        print(l)

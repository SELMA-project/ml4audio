import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, ClassVar, Any, Annotated

from beartype.vale import Is

from data_io.download_extract_files import wget_file, extract_file
from data_io.readwrite_files import (
    read_csv,
)
from misc_utils.cached_data import (
    CachedData,
)
from misc_utils.dataclass_utils import (
    UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix

split_names: ClassVar[list[str]] = ["train", "dev", "test"]


@dataclass
class CommonVoiceDatum:
    client_id: str
    path: str
    sentence: str
    up_votes: int
    down_votes: int
    age: str
    gender: str
    accents: list[str]
    locale: str
    segment: Any

    @staticmethod
    def from_dict(d: dict):
        d["accents"] = list(d["accents"].split(","))
        return CommonVoiceDatum(**d)


@dataclass
class CommonVoiceExtracted(CachedData):
    """
    data_dir look like:
          9994240 Jul 23 09:37 clips/
           819191 Jun 27 23:16 dev.tsv
           835181 Jun 27 23:16 invalidated.tsv
         21018325 Jun 27 23:16 other.tsv
             8314 Jul  4 15:33 reported.tsv
           805125 Jun 27 23:16 test.tsv
          1039550 Jun 27 23:16 train.tsv
         10396147 Jun 27 23:16 validated.tsv

    "validated" is superset of train+test, it is NOT a dev set!!
    """

    url: str = UNDEFINED
    cache_base: PrefixSuffix = PrefixSuffix("base_path", f"data/ASR_DATA/COMMON_VOICE")
    use_hash_suffix: ClassVar[bool] = False
    SPLITS: ClassVar[list[str]] = ["train", "test"]  # validated is NOT a dev-set

    @property
    def name(self):
        name = self.url.split("/")[-1].split(".tar.gz")[0]
        return name

    @property
    def lang(self):
        # TODO is this always working?
        return self.name.split("-")[-1]

    @property
    def data_dir(self):
        s = self.name.replace(f"-{self.lang}", "")
        return f"{self.cache_dir}/{s}/{self.lang}"

    @property
    def clips_dir(self):
        return f"{self.data_dir}/clips"

    def _download_and_extract_raw_data(self):
        wget_file(f'"{self.url}"', str(self.cache_dir), verbose=True)
        extract_folder = str(self.cache_dir)
        file = next(Path(extract_folder).glob(f"{self.name}*"))
        extract_file(
            f'"{file}"', extract_folder, lambda dirr, file: f"tar xzf {file} -C {dirr}"
        )
        os.remove(file)

    def _build_cache(self) -> None:
        self._download_and_extract_raw_data()

    def get_split_data(
        self, split_name: Annotated[str, Is[lambda s: s in CommonVoiceExtracted.SPLITS]]
    ) -> Iterator[CommonVoiceDatum]:
        csv_file = f"{self.data_dir}/{split_name}.tsv"
        for d in read_csv(csv_file, use_json_loads=False):
            yield CommonVoiceDatum.from_dict(d)


if __name__ == "__main__":

    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")

    url = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-10.0-2022-07-04/cv-corpus-10.0-2022-07-04-ur.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3KSV4BYWU%2F20220723%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220723T081134Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEDkaDF5YqFAal1ZS6w%2BgtyKSBKmFU%2FrVpro0cz6DldEHY7ruoMwdqSM%2BX%2B6yv9YdipJtgYf35e7WK4GTooI3JtifLgkqYG%2BQkOPARw27TI99d4hw24ocTXyh1u1O0MGF%2BmjJhJoO6YwJmFAvmS8f%2F6T0sVI4%2FAYknzzh834zm5JgjfdLtMi8UQR%2B2MvaM5HxtlP0evmEbRyDHv9LHE%2BykU69VPko%2FDzE5lte668FaVpQ%2FHRkBREaH09qkhd3WnpxwNzPeaqWnkIPZFnl0gNh1d3K0NaBklWHlSGt%2FJNDMlEiQH3qs7F1bxt%2FkZXEny3H3BkElG4VLRNEIMIXSJ7%2BHxM34pEHqzz%2FrYou%2B2zaHVg0tea3hE7Ue2%2FZHfhtYh2YuYDhW23TX8V4xcaDAECG6a68JYhM7VLPUoOQVJX4h%2FC6%2BkrIDiR7%2B3CEDngOghPbn%2FR9qFAXVHKTIzIostt80u1O%2FR%2BoHTbnJ5OJXXu6kD3TJC%2B0sgSu2Lu2vLryCSZPdypLiP9a7Vggi58nMEBbsKdZshCjY5sa6uxsEfFNDx%2BW59UAPrUmX97xAodoxMRk0gsPrv2%2BoODflngkPJtUkKyVsL51zwjxWkfEeWVQEFCB2YNdklH4IvN6X09D2ZGz79hY1KrA112puGgZdZFkXU%2F2NvTUnNshJghdpZgp15kFlJ1kryfN8XiCxMfWEYw34hER84MZUQ%2FLrid%2Fs9IJ%2FX7STxcuKP3J7pYGMip9LF%2FjdkEfQQs2nM4ItGt6OHT3IFe1Ef%2Fswy99OgYD33wZuDI0ReDoSuc%3D&X-Amz-Signature=d688a4078332d7aa2f2d47d5368c91d7598bbfa0c3195162884fe53e402f0771&X-Amz-SignedHeaders=host"
    corpus: CommonVoiceExtracted = CommonVoiceExtracted(
        url=url, overwrite_cache=True
    ).build()

    for o in itertools.islice(corpus.get_split_data("train"), 10):
        print(o)

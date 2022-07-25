import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, ClassVar, Any, Annotated, Optional

from beartype.vale import Is

from data_io.download_extract_files import wget_file, extract_file
from data_io.readwrite_files import (
    read_csv,
)
from misc_utils.buildable import Buildable
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
class CommonVoiceExtracted(Buildable):
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

    url: Optional[str] = field(repr=False, default=None)
    data_base: PrefixSuffix = PrefixSuffix("base_path", f"data/ASR_DATA/COMMON_VOICE")
    SPLITS: ClassVar[list[str]] = ["train", "test"]  # validated is NOT a dev-set
    targz_file: Optional[str] = None
    _name: str = field(init=False, repr=True)

    def __post_init__(self):
        if self.url is not None:
            self._name = self.url.split("/")[-1].split(".tar.gz")[0]
        elif self.targz_file is not None:
            self._name = self.targz_file.split("/")[-1].split(".tar.gz")[0]
        else:
            raise AssertionError("either url or targz_file is needed!")

    @property
    def name(self):
        return self._name

    @property
    def lang(self):
        # TODO is this always working?
        return self.name.split("-")[-1]

    @property
    def data_dir(self):
        return f"{self.data_base}/{self.name}/{self.version}/{self.lang}"

    @property
    def version(self):
        return "-".join(self.name.split("-")[:-1])

    @property
    def clips_dir(self):
        return f"{self.data_dir}/clips"

    def _download_and_extract_raw_data(self):
        extract_folder = f"{self.data_base}/{self.name}"
        os.makedirs(extract_folder, exist_ok=True)

        if self.targz_file is None:
            wget_file(f'"{self.url}"', extract_folder, verbose=True)
            file = next(Path(extract_folder).glob(f"{self.name}*"))
        else:
            file = self.targz_file

        print(f"extracting: {self.name}")
        extract_file(
            f'"{file}"', extract_folder, lambda dirr, file: f"tar xzf {file} -C {dirr}"
        )

        if self.targz_file is None:
            os.remove(file)

    @property
    def _is_ready(self) -> bool:
        return os.path.isdir(self.clips_dir) and all(
            os.path.isfile(f"{self.data_dir}/{s}") for s in self.SPLITS
        )

    def _build_self(self) -> Any:
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

    url = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-10.0-2022-07-04/cv-corpus-10.0-2022-07-04-de.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3NUOIJFM4%2F20220723%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220723T092446Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEDoaDHWugLCAcPgojreCESKSBAvSpfh%2BvzRbnLXAeqnyOr4iXCvpwutOFIAfhz%2FqipevlQwjR2A1mYjIJugei7A5NprV8jlhNbDUJegGjh%2B7TMVldauTyysyVAL8eY1QZoN0KEDVVPEbQwV2mdLF7K63dyyt15G5daqu8W3KgjL9KMftUpHPj2DK39vLBotYDzoGT3AHHMSmdhaW%2B5oqfdQZeoMDhwKhnFZpHgBT6VaEWf3nRL5ipz7aGZKHURRR6%2B%2FPFV2ajB1DXKFDSBdav9RgHUYbnMbCor%2FVoDurfoIS7CgyyMQxtwJZJv18iHK4ZZCbiAaYdq1L07Gr0ytzmAqu8Vc3qlyees2ou1b6a3AtU%2BJpyX%2Bj2borKdw0vPOC%2Bk%2BPkUvazFkP%2BEj%2FvlMokh%2FDqvh6w7by0UQt5WbAWIh%2Bl7bgEG0xQ1Gj4hXTeFkPPpkhw8GC4%2B7XhekyWBjIpf%2FyC9IEGLBSoUFY8f5g3mbYfcxlAXYXqW3tpYqxXSDsEetIyl%2BMG5i6XFKEZX24y%2FbTJmvbYDHf%2BpcCw4FGSkDP1xCF25OvXb6WWT7qCXo3%2FOTBYetukE19oU2UDSo4pQc3oBj%2BwROP8vT15JnDNtnSccfi7k45Nw11clN5J2EuliurJbnQAErWsdrM98k%2BGapjkGFWecve5FZRaB8ka8AZDJiHWg0RNiIl5WH9ux5qqZMiynNSxDtAKNqDLyLCQx4yi6SwKPnw7pYGMir2WbNNrKZG0oo3cZz50kELWsu7Iy8E%2BQteofOwj6wtuoxL9D2ftXGudn0%3D&X-Amz-Signature=e25ce4852a38d15d1172b0f016f72e63825d5396796500bd4a1d3a064246a019&X-Amz-SignedHeaders=host"
    corpus: CommonVoiceExtracted = CommonVoiceExtracted(
        targz_file=f"{BASE_PATHES['base_path']}/data/ASR_DATA/COMMON_VOICE/cv-corpus-10.0-2022-07-04-es.tar.gz",
        url=None,
    )
    assert os.path.isdir(corpus.data_dir), corpus.data_dir
    corpus.build()

    for o in itertools.islice(corpus.get_split_data("train"), 10):
        print(o)

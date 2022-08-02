import itertools
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, ClassVar, Any, Annotated, Optional

from beartype.vale import Is

from data_io.download_extract_files import wget_file, extract_file
from data_io.readwrite_files import (
    read_csv,
)
from misc_utils.buildable import Buildable
from misc_utils.buildable_data import BuildableData
from misc_utils.dataclass_utils import UNDEFINED
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.audio_utils.audio_data_models import AudioTextData, ArrayText
from ml4audio.audio_utils.torchaudio_utils import load_resample_with_torch

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
        d["up_votes"] = int(d["up_votes"])
        d["down_votes"] = int(d["down_votes"])
        return CommonVoiceDatum(**d)


@dataclass
class CommonVoiceExtracted(BuildableData):
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

    url: Optional[str] = field(repr=True, default=None)
    base_dir: PrefixSuffix = PrefixSuffix("base_path", f"data/ASR_DATA/COMMON_VOICE")
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
    def extract_dir(self):
        return f"{self.data_dir}/{self.version}/{self.lang}"

    @property
    def version(self):
        return "-".join(self.name.split("-")[:-1])

    @property
    def clips_dir(self):
        return f"{self.extract_dir}/clips"

    @property
    def _is_data_valid(self) -> bool:
        return os.path.isdir(self.clips_dir) and all(
            os.path.isfile(f"{self.extract_dir}/{s}.tsv") for s in self.SPLITS
        )

    def _build_data(self) -> Any:
        self._download_and_extract_raw_data()

    def _download_and_extract_raw_data(self):
        """
        # might be necessary to extract tar.gz here, somehow this raid that I am using does deadlocks/hangs forever when trying to extract big files into it
        """
        with TemporaryDirectory(prefix="common_voice_extract") as tmp_dir:
            extract_folder = tmp_dir
            print(f"{extract_folder=}")
            os.makedirs(extract_folder, exist_ok=True)

            if self.targz_file is None:
                wget_file(f'"{self.url}"', extract_folder, verbose=True)
                file = next(Path(extract_folder).glob(f"{self.name}*"))
            else:
                file = self.targz_file

            print(f"extracting: {self.name}")
            extract_file(
                f'"{file}"',
                extract_folder,
                lambda dirr, file: f"tar xzf {file} -C {dirr}",
            )

            shutil.copytree(
                extract_folder, f"{self.base_dir}/{self.name}", dirs_exist_ok=True
            )

            if self.targz_file is None:
                os.remove(file)

    def get_split_data(
        self, split_name: Annotated[str, Is[lambda s: s in CommonVoiceExtracted.SPLITS]]
    ) -> Iterator[CommonVoiceDatum]:
        csv_file = f"{self.extract_dir}/{split_name}.tsv"
        for d in read_csv(csv_file, use_json_loads=False):
            yield CommonVoiceDatum.from_dict(d)


@dataclass
class CommonVoiceAuteda(AudioTextData, Buildable):
    raw_data: CommonVoiceExtracted = UNDEFINED
    split_name: str = UNDEFINED
    sample_rate: int = 16000

    @property
    def name(self) -> str:
        return f"{self.raw_data.name}-{self.split_name}"

    def __iter__(self) -> Iterator[ArrayText]:
        for d in self.raw_data.get_split_data(self.split_name):
            array = load_resample_with_torch(
                data_source=f"{self.raw_data.clips_dir}/{d.path}",
                target_sample_rate=self.sample_rate,
            ).numpy()
            yield array, d.sentence


if __name__ == "__main__":

    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")

    urls = [
        "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-10.0-2022-07-04/cv-corpus-10.0-2022-07-04-fr.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3J5BBMF42%2F20220801%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220801T190622Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEBwaDDeo96fG%2FP4o7jBCDSKSBNcK7WPx%2FL7bMv6RnysJOa7q7IAzSdUZe%2BzcSH8cv0q0S4uaXtVhrQk9Ciq2goWsLSSlOcXTB%2BNxxRXQUw%2Br%2BQQZmAy0XcYZVo%2Fg1Ex3f%2B7hrduJKpKuS0sSuDTzYEFYF2aVX0vlQ%2B6LmBrVC8xF7Q%2B%2BqsC7LjPExYrjp1aqOK7zxNk2VXzQp5hrzWICgfFD0k4ITy%2Bk%2B6GhJvnild%2BSE0fkPlQU6krhrXuJtjVuJfzj03qu%2B9E0%2FFBpSOmW9JtoVBUz0n6Zuek2RV%2Bp4cPgi8XSXzrZj8NaF5lQQln7z8uHBCJi9iMao7gAOBPjik3eXw%2BdKV9AEqVYIzOXFKVHKAeFcYNuYXg%2FfffF6kAcHQN9LI5tnOt0CIrjgvp68ebOmDw18heymBYf6ptLKYEYe0dRZykfnAsxJPg8GPcjuIoQV1wySr9Zx6syhqdJ4uqDRz15sXPgO1zBg21VC0fBZ6s58W2jnVAfHLEiE3I8nZUHvs%2BQ1a7cF6TVuRRIAr6RX8emdvCqe5C1ZBea%2FFa2FNKSI%2F5qkuvS4i1Cb2JR6FlxfHn8QwVFRmdJdidtuNt6shcyg%2BIjfN4MHpzCzIE3KRn8F%2BU62B0SiclP5mfTUeZ30VYEZp%2ByWyB3HC6ATNSKEjwndc0pwk87NHpgsjrXGsL7dCOrOu4Pe6O62KaMPLFs4mRWHO7vWt1wdnhC2l9Bk4zOKP%2FGoJcGMip548b0zOthkp0dX1kVcJBxP%2ForTAqQmQpFB%2FFs3vbJTxn93BceNg8QTbE%3D&X-Amz-Signature=424e9b178b2a14a41045a0c213d02392e879b4ed0bf4a56bb78868d0f275a743&X-Amz-SignedHeaders=host"
    ]
    for url in urls:
        corpus: CommonVoiceExtracted = CommonVoiceExtracted(
            # targz_file=f"{BASE_PATHES['base_path']}/data/ASR_DATA/COMMON_VOICE/cv-corpus-10.0-2022-07-04-es.tar.gz",
            url=url,
        )
        corpus.build()

        for o in itertools.islice(corpus.get_split_data("train"), 10):
            print(o)

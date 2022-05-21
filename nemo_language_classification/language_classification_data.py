# pylint: skip-file
# pylint: disable-all
import json
from typing import Optional
from urllib.request import urlopen

import sys

from data_io.readwrite_files import write_jsonl
from ml4audio.audio_utils.torchaudio_utils import torchaudio_info

sys.path.append("")

from itertools import islice
from random import shuffle

import os
import shutil

import datasets
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000


def lang_clf_nemo_datum(d) -> Optional[dict]:
    file = d["path"]
    if os.path.isfile(file):

        # x, _sr = librosa.load(file, sr=TARGET_SAMPLE_RATE)
        # duration=librosa.get_duration(x, sr=_sr)
        num_frames, sample_rate, duration = torchaudio_info(file)
        return {
            "audio_filepath": file,
            "duration": duration,
            "label": d["locale"],
            "text": "_",  # d["sentence"]
            "offset": 0.0,
        }
    else:
        return None


# def copy_data(d):
#     shutil.copy(d["audio_filepath"], corpus_dir)


def rm_mkdir(dirr):
    if os.path.isdir(dirr):
        shutil.rmtree(dirr)
    os.makedirs(dirr)


assert (
    "HF_DATASETS_CACHE" in os.environ
), f'do: export HF_DATASETS_CACHE="/path/to/another/directory"'
assert "HF_HOME" in os.environ, f'do export HF_HOME="/somewhere/path/huggingface_cache"'

# export HF_DATASETS_CACHE={os.environ['BASE_PATH']}/huggingface_cache/datasets
# export HF_HOME={os.environ['BASE_PATH']}/huggingface_cache

if __name__ == "__main__":
    cv_info_json = urlopen(
        "https://huggingface.co/datasets/common_voice/raw/main/dataset_infos.json"
    ).read()
    cv_info = json.loads(cv_info_json)
    cv_languages = cv_info.keys()
    print(f"{cv_languages=}")
    # assert False

    num_samples_per_lang = 10_000
    manifests_dir = f"{os.environ['BASE_PATH']}/data/AUDIO_DATA/lang_clf_data_7lanuages"
    rm_mkdir(manifests_dir)

    # it = iter(_LANGUAGES.keys())
    languages_of_interest = ["en", "de", "es", "ru", "pt", "fr", "it"]

    for lang in languages_of_interest:

        for split_name in ["train", "validation", "test"]:
            try:
                ttm = num_samples_per_lang * 10  # so ttm == ten times more
                ds_ttm = datasets.load_dataset(
                    "common_voice",
                    lang,
                    keep_in_memory=True,
                    split=f"{split_name}[:{ttm}]",
                )
            except:
                ds_ttm = datasets.load_dataset(
                    "common_voice", lang, keep_in_memory=True, split=f"{split_name}"
                )

            data_ttm = list(
                filter(
                    lambda x: x is not None, (lang_clf_nemo_datum(d) for d in ds_ttm)
                )
            )
            shuffle(
                data_ttm
            )  # not sure whether common-voice data is already shuffled? I don't want only few speakers!
            data = list(islice(data_ttm, num_samples_per_lang))
            # for d in data:
            #     copy_data(d)

            write_jsonl(
                f"/{manifests_dir}/{split_name}_manifest.jsonl",
                tqdm(data, desc=f"{lang}: {split_name}"),
                mode="ab",
            )

import os
from data_io.readwrite_files import read_jsonl
from ml4audio.audio_utils.audio_io import read_audio_chunks_from_file
from functools import partial
from itertools import groupby
from itertools import islice
from typing import Dict, List, Tuple

from nemo_language_classification.common import LangClf
from nemo_language_classification.nemo_lang_clf import NemoLangClf

from pprint import pprint

from sklearn import metrics
from tqdm import tqdm


def get_data(
    val_manifest=f"{os.environ['BASE_PATH']}/data/lang_clf_data/validation_manifest.jsonl",
):
    data = list(read_jsonl(val_manifest))
    lang2data = {
        k: list(islice(g, 10))
        for k, g in groupby(
            sorted(data, key=lambda x: x["label"]), lambda x: x["label"]
        )
    }
    lang2data = {"en": lang2data["en"], "de": lang2data["de"]}
    data = [d for g in lang2data.values() for d in g]
    return data


def get_max(o: Dict[str, float]):
    label, value = max(o.items(), key=lambda x: x[1])
    return label


def benchmark_lang_clf(mdl: LangClf, input_output: List[Tuple[str, str]]):
    mdl.init()
    sr = 16000
    chunk_dur = 8
    id_pred_targets = [
        (f"{wav_file}-{k}", get_max(mdl.predict(chunk)), target)
        for wav_file, target in tqdm(input_output)
        for k, chunk in enumerate(
            read_audio_chunks_from_file(wav_file, sr, chunk_duration=chunk_dur)
        )
        if len(chunk) > (chunk_dur / 2) * sr
    ]
    eids, preds, targets = (list(x) for x in zip(*id_pred_targets))
    print(f"labels: {targets}")
    clf_report = metrics.classification_report(
        y_true=targets,
        y_pred=preds,
        # labels=target_names,
        digits=3,
        output_dict=True,
    )
    return clf_report


if __name__ == "__main__":

    # val_manifest=f"{os.environ['BASE_PATH']}/data/lang_clf_data/train_manifest.jsonl"
    # model_file="/nm-raid/nishome/thimmelsba/iais_code/SLYTHERIN/end2end-asr/nemo_experiments/SpeakerNet/2021-07-22_12-01-52/checkpoints/SpeakerNet--val_loss=8.97-epoch=0-last.ckpt"
    # "{os.environ['BASE_PATH']}/results/TRAINING/LANG_CLF/debug/SpeakerNet/2021-07-23_10-14-04/checkpoints/SpeakerNet--val_loss=6.84-epoch=1-last.ckpt"

    data = get_data()
    input_output = [
        (
            d["audio_filepath"].replace("data/huggingface/", "huggingface_cache/"),
            d["label"],
        )
        for d in data
    ]
    benchmark_fun = partial(benchmark_lang_clf, input_output=input_output)
    pprint(
        benchmark_fun(
            NemoLangClf(
                model_file=f"{os.environ['BASE_PATH']}/results/TRAINING/LANG_CLF/debug/SpeakerNet/2021-07-23_10-14-04/checkpoints/SpeakerNet--val_loss=6.84-epoch=1-last.ckpt"
            )
        )
    )
    # pprint(benchmark_fun(Wav2vecPyctcLangClf()))

"""
python nemo_language_classification/benchmark_lang_clf.py "${BASE_PATH}/results/TRAINING/LANG_CLF/debug/SpeakerNet/2021-07-23_10-14-04/checkpoints/SpeakerNet--val_loss=6.84-epoch=1-last.ckpt" "{os.environ['BASE_PATH']}/data/lang_clf_data/validation_manifest.jsonl"
"""

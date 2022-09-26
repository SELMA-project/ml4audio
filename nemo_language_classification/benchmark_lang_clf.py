import os

from beartype import beartype

from data_io.readwrite_files import read_jsonl
from misc_utils.beartypes import NeList
from ml4audio.audio_utils.audio_io import read_audio_chunks_from_file
from functools import partial
from itertools import groupby
from itertools import islice
from typing import Dict, List, Tuple

from nemo_language_classification.nemo_lang_clf import NemoLangClf

from pprint import pprint

from sklearn import metrics
from tqdm import tqdm


def get_data(
    val_manifest=f"{os.environ['BASE_PATH']}/data/lang_clf_data/validation_manifest.jsonl",
):
    data = list(read_jsonl(val_manifest))
    lang2data = {
        k: list(islice(g, 100))
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


@beartype
def benchmark_lang_clf(
    mdl: NemoLangClf, wavfile_label: NeList[tuple[str, str]]
) -> dict:
    sr = 16000
    chunk_dur = 8
    id_pred_targets = [
        (f"{wav_file}-{k}", get_max(mdl.predict(chunk)), target)
        for wav_file, target in tqdm(wavfile_label)
        for k, chunk in enumerate(
            read_audio_chunks_from_file(wav_file, sr, chunk_duration=chunk_dur)
        )
        if len(chunk) > (chunk_dur / 2) * sr
    ]
    eids, preds, targets = (list(x) for x in zip(*id_pred_targets))
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
    # model_file="{BASE_PATH}/.../end2end-asr/nemo_experiments/SpeakerNet/2021-07-22_12-01-52/checkpoints/SpeakerNet--val_loss=8.97-epoch=0-last.ckpt"
    # "{os.environ['BASE_PATH']}/results/TRAINING/LANG_CLF/debug/SpeakerNet/2021-07-23_10-14-04/checkpoints/SpeakerNet--val_loss=6.84-epoch=1-last.ckpt"

    data = get_data(
        val_manifest=f"{os.environ['BASE_PATH']}/data/AUDIO_DATA/lang_clf_data_7lanuages/test_manifest.jsonl"
    )
    input_output = [
        (
            d["audio_filepath"].replace("data/huggingface/", "huggingface_cache/"),
            d["label"],
        )
        for d in data
    ]
    pprint(
        benchmark_lang_clf(
            NemoLangClf(
                model_file=f"{os.environ['BASE_PATH']}/iais_code/ml4audio/nemo_experiments/titanet-finetune-lang-clf/2022-09-25_13-12-59/checkpoints/titanet-finetune-lang-clf.nemo"
            ).build(),
            wavfile_label=input_output,
        )
    )
    # pprint(benchmark_fun(Wav2vecPyctcLangClf()))


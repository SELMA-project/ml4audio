import json
import os

import matplotlib.pyplot
import wget

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import (
    rttm_to_labels,
    labels_to_pyannote_object,
)
from pyannote.core import Annotation
from omegaconf import OmegaConf

# see: https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb


def get_test_data(data_dir: str):
    an4_audio = os.path.join(data_dir, "an4_diarize_test.wav")
    an4_rttm = os.path.join(data_dir, "an4_diarize_test.rttm")
    if not os.path.exists(an4_audio):
        an4_audio_url = (
            "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
        )
        an4_audio = wget.download(an4_audio_url, data_dir)
    if not os.path.exists(an4_rttm):
        an4_rttm_url = (
            "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
        )
        an4_rttm = wget.download(an4_rttm_url, data_dir)
    return an4_audio, an4_rttm


def create_test_manifest(data_dir, an4_audio, an4_rttm):
    meta = {
        "audio_filepath": an4_audio,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": 2,  # TODO: WTF!! what if I don't know how many speakers?
        "rttm_filepath": an4_rttm,
        "uem_filepath": None,
    }
    with open(f"{data_dir}/input_manifest.json", "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")


def get_test_config(data_dir):
    # MODEL_CONFIG = os.path.join(data_dir, "offline_diarization.yaml")
    # if not os.path.exists(MODEL_CONFIG):
    #     config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization.yaml"
    #     MODEL_CONFIG = wget.download(config_url, data_dir)

    MODEL_CONFIG = "offline_diarization.yaml"
    config = OmegaConf.load(MODEL_CONFIG)
    print(OmegaConf.to_yaml(config))
    return config


if __name__ == "__main__":

    ROOT = os.getcwd()
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    an4_audio, an4_rttm = get_test_data(data_dir)

    labels = rttm_to_labels(an4_rttm)
    reference: Annotation = labels_to_pyannote_object(labels)
    print(f"{labels=},{reference=}")
    create_test_manifest(data_dir, an4_audio, an4_rttm)
    config = get_test_config(data_dir)

    output_dir = os.path.join(ROOT, "outputs")

    pretrained_vad = "vad_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.diarizer.manifest_filepath = "data/input_manifest.json"
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
    # config.diarizer.speaker_embeddings.oracle_vad_manifest = None

    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = True

    # Here we use our inhouse pretrained NeMo VAD
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.window_length_in_sec = 0.15
    config.diarizer.vad.shift_length_in_sec = 0.01
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.min_duration_on = 0.1
    config.diarizer.vad.parameters.min_duration_off = 0.4

    sd_model = ClusteringDiarizer(cfg=config)

    sd_model.diarize()

    from nemo.collections.asr.parts.utils.vad_utils import plot

    # plot(
    #     an4_audio,
    #     f"{output_dir}/vad_outputs/overlap_smoothing_output_median_0.875/an4_diarize_test.median",
    #     an4_rttm,
    #     per_args=config.diarizer.vad.parameters,  # threshold
    # )
    # matplotlib.pyplot.show()

    print(f"postprocessing_params: {config.diarizer.vad.parameters}")

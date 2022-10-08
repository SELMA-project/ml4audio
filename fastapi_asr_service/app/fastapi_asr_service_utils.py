import os
from pathlib import Path

from omegaconf import OmegaConf

from data_io.readwrite_files import read_json
from misc_utils.dataclass_utils import decode_dataclass
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from nemo_vad.nemo_offline_vad import NemoOfflineVAD


def load_asr_inferencer():
    cache_root_in_container = os.environ["CACHE_ROOT"]
    cache_root = os.environ.get("cache_root", cache_root_in_container)
    BASE_PATHES["base_path"] = "/"
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["asr_inference"] = PrefixSuffix("cache_root", "ASR_INFERENCE")
    BASE_PATHES["am_models"] = PrefixSuffix("cache_root", "AM_MODELS")
    p = next(
        Path(cache_root).rglob("Aschinglupi*/dataclass.json")
    )  # TODO(tilo): hard-coded the class-name here!!
    jzon = read_json(str(p))
    inferencer = decode_dataclass(jzon)
    inferencer.build()
    return inferencer


default_vad_config = {
    "name": "vad_inference_postprocessing",
    "dataset": None,
    "num_workers": 0,
    "sample_rate": 16000,
    "gen_seg_table": True,
    "write_to_manifest": True,
    "prepare_manifest": {"auto_split": True, "split_duration": 400},
    "vad": {
        "model_path": "app/vad_multilingual_marblenet.nemo",
        "parameters": {
            "normalize_audio": False,
            "window_length_in_sec": 0.15,
            "shift_length_in_sec": 0.01,
            "smoothing": "median",
            "overlap": 0.875,
            "postprocessing": {
                "onset": 0.3,
                "offset": 0.2,
                "pad_onset": 0.1,
                "pad_offset": 0.1,
                "min_duration_on": 0.5,
                "min_duration_off": 1.0,
                "filter_speech_first": True,
            },
        },
    },
    "prepared_manifest_vad_input": None,
    "frame_out_dir": "vad_frame",
    "smoothing_out_dir": None,
    "table_out_dir": None,
    "out_manifest_filepath": None,
}


def load_vad_inferencer() -> NemoOfflineVAD:
    cfg = OmegaConf.create(default_vad_config)
    cfg.vad.parameters.window_length_in_sec = 0.15
    cfg.vad.parameters.postprocessing.onset = 0.1
    cfg.vad.parameters.postprocessing.offset = 0.05
    cfg.vad.parameters.postprocessing.min_duration_on = 0.1
    cfg.vad.parameters.postprocessing.min_duration_off = 3.0
    cfg.vad.parameters.smoothing = "median"
    cfg.vad.parameters.overlap = 0.875
    vad = NemoOfflineVAD(cfg)
    vad.build()
    return vad

import os
from pathlib import Path

from omegaconf import OmegaConf

from data_io.readwrite_files import read_json
from misc_utils.dataclass_utils import decode_dataclass
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from nemo_vad.nemo_offline_vad import NemoOfflineVAD


def load_asr_inferencer():
    cache_root_in_container = "/model"
    cache_root = os.environ.get("cache_root", cache_root_in_container)
    BASE_PATHES["base_path"] = "/"
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["asr_inference"] = PrefixSuffix("cache_root", "ASR_INFERENCE")
    BASE_PATHES["am_models"] = PrefixSuffix("cache_root", "AM_MODELS")
    p = next(Path(cache_root).rglob("HfAsrPipeline*/dataclass.json"))
    jzon = read_json(str(p))
    inferencer = decode_dataclass(jzon)
    inferencer.build()
    return inferencer


def load_vad_inferencer() -> NemoOfflineVAD:
    config_yaml = "app/vad_inference_postprocessing.yaml"
    # config_yaml = "speaker_tasks/tests/vad_test_config.yaml"
    cfg = OmegaConf.load(config_yaml)
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

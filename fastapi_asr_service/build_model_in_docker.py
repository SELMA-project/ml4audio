import os
from pathlib import Path
from pprint import pprint

from data_io.readwrite_files import read_json
from misc_utils.dataclass_utils import (
    decode_dataclass, to_dict,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.asr_inference.hf_asr_pipeline import (
    HfAsrPipelineFromLogitsInferencerDecoder,
)

if __name__ == "__main__":
    """
    maybe it acts as kind of sanity/integration test??
    """

    cache_root_in_container = "/model"
    cache_root = os.environ.get("cache_root", cache_root_in_container)
    BASE_PATHES["base_path"] = "/"
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["asr_inference"] = PrefixSuffix("cache_root", "ASR_INFERENCE")
    BASE_PATHES["am_models"] = PrefixSuffix("cache_root", "AM_MODELS")

    p = next(Path(cache_root).rglob("HfAsrPipeline*/dataclass.json"))

    jzon = read_json(str(p))
    inferencer: HfAsrPipelineFromLogitsInferencerDecoder = decode_dataclass(jzon)
    inferencer.build()
    pprint(to_dict(inferencer))

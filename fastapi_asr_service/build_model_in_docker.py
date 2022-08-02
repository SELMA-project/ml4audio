import json
import os
from pathlib import Path

from audiomonolith.asr_inference.asr_chunk_infer_glue_pipeline import Aschinglupi
from data_io.readwrite_files import read_file, read_json
from misc_utils.dataclass_utils import (
    deserialize_dataclass,
    dataclass_to_yaml,
    decode_dataclass,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix

if __name__ == "__main__":
    """
    once this was necessary to build the binary kenlm, currenlty I am NOT using binary but simply the arpa!
    not sure what this is good for!

    maybe it acts as kind of sanity/integration test??
    """

    cache_root_in_container = "/code/model"
    cache_root = os.environ.get("cache_root", cache_root_in_container)
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["asr_inference"] = PrefixSuffix("cache_root", "ASR_INFERENCE")
    p = next(Path(cache_root).rglob("Aschinglupi*/dataclass.json"))

    jzon = read_json(str(p))
    inferencer: Aschinglupi = decode_dataclass(jzon)
    print(f"{inferencer.hf_asr_decoding_inferencer.logits_inferencer=}")
    inferencer.build()
    print(dataclass_to_yaml(inferencer))

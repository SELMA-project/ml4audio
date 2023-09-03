import os
import shutil
import tempfile

import numpy as np
from beartype import beartype
from omegaconf import OmegaConf

from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.audio_utils.audio_io import load_resample_with_soundfile, ffmpeg_load_trim
from nemo_vad.nemo_offline_vad import NemoOfflineVAD
from nemo_vad.tests.vad_infer_almost_original import (
    nemo_offline_vad_infer_main_original,
)
import logging

from conftest import get_test_cache_base

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

# fmt: off
# used vad_infer_almost_original.py to create this expected
expected = [(0.31, 2.93), (3.27, 6.109999999999999), (6.81, 9.83), (10.69, 13.149999999999999), (13.69, 16.35), (17.21, 19.23), (19.54, 20.45), (21.37, 24.37)]
# fmt: on

BASE_PATHES["cache_root"] = get_test_cache_base()


@beartype
def vad_assertions(start_ends: list[tuple[float, float]]):
    assert len(start_ends) == len(expected), f"{len(start_ends)=},{len(expected)=}"
    starts, ends = [np.asarray(x) for x in zip(*start_ends)]
    starts_exp, ends_exp = [np.asarray(x) for x in zip(*expected)]
    print(f"{starts=},{ends=}")
    print(f"{starts_exp=},{ends_exp=}")
    # assert start_ends==expected
    assert np.allclose(starts, starts_exp, atol=1e-2)
    assert np.allclose(ends, ends_exp, atol=1e-2)


# for parameters see: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/asr/conf/vad/vad_inference_postprocessing.yaml
default_vad_config = {
    "name": "vad_inference_postprocessing",
    "dataset": None,
    "num_workers": 0,
    "sample_rate": 16000,
    "gen_seg_table": True,
    "write_to_manifest": True,
    "prepare_manifest": {"auto_split": True, "split_duration": 400},
    "vad": {
        "model_path": "vad_marblenet",
        "parameters": {
            "normalize_audio": False,
            "window_length_in_sec": 0.15,
            "shift_length_in_sec": 0.01,
            "smoothing": "median",
            "overlap": 0.875,
            "postprocessing": {
                "onset": 0.4,
                "offset": 0.7,  # TODO(tilo): makes no sense to me
                "pad_onset": 0.05,
                "pad_offset": -0.1,
                "min_duration_on": 0.2,
                "min_duration_off": 0.2,
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

# TODO: the test is broken!
def test_nemo_offline_vad(
    librispeech_audio_file,
):
    audio = ffmpeg_load_trim(librispeech_audio_file)

    vad = NemoOfflineVAD(name="test-vad", cfg=default_vad_config)
    vad.build()
    with vad:
        start_ends, _ = vad.predict(audio)
        vad_assertions(start_ends)


# def test_nemo_original_vad(
#     librispeech_audio_file,
#     config_yaml="tests/resources/vad_inference_postprocessing_original.yaml",
# ):
#
#     with tempfile.TemporaryDirectory() as tmpdir:
#         start_ends = nemo_offline_vad_infer_main_original(
#             audio_file=librispeech_audio_file,
#             config_yaml=config_yaml,
#             data_dir=tmpdir,
#         )
#         shutil.rmtree("vad_frame_for_testing", ignore_errors=True)
#     vad_assertions(start_ends)

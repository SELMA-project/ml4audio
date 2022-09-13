import shutil
import tempfile

import numpy as np
from beartype import beartype
from omegaconf import OmegaConf

from nemo_vad.nemo_offline_vad import NemoOfflineVAD
from nemo_vad.tests.vad_infer_almost_original import (
    nemo_offline_vad_infer_main_original,
)
import logging

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

# fmt: off
# used vad_infer_almost_original.py to create this expected
expected = [(0.31, 2.93), (3.27, 6.109999999999999), (6.81, 9.83), (10.69, 13.149999999999999), (13.69, 16.35), (17.21, 19.23), (19.54, 20.45), (21.37, 24.37)]
# fmt: on


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


def test_nemo_offline_vad(
    audio_file="tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.wav",
    config_yaml="nemo_vad/tests/vad_test_config.yaml",
):
    cfg = OmegaConf.load(config_yaml)
    vad = NemoOfflineVAD(cfg)
    vad.build()
    start_ends, _ = vad.predict(audio_file)
    vad_assertions(start_ends)


def test_nemo_original_vad(
    audio_file="tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.wav",
    config_yaml="nemo_vad/tests/vad_test_config.yaml",
):

    with tempfile.TemporaryDirectory() as tmpdir:
        start_ends = nemo_offline_vad_infer_main_original(
            audio_file=audio_file,
            config_yaml=config_yaml,
            data_dir=tmpdir,
        )
        shutil.rmtree("vad_frame_for_testing", ignore_errors=True)
    vad_assertions(start_ends)

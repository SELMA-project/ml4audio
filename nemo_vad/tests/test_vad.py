import numpy as np
import pytest
from numpy.testing import assert_allclose

from ml4audio.audio_utils.audio_io import (
    load_and_resample_16bit_PCM,
    break_array_into_chunks,
)
from nemo_vad.nemo_streaming_vad import NeMoVAD

frame_duration = 0.1

file = "nemo_vad/tests/resources/VAD_demo.wav"
SR = 16_000


@pytest.fixture()
def vad():
    return NeMoVAD(
        threshold=0.3,
        frame_duration=frame_duration,
        window_len_in_secs=4 * frame_duration,
        input_sample_rate=SR,
    ).build()


@pytest.fixture()
def audio_arrays():
    speech_array = load_and_resample_16bit_PCM(file, SR)
    arrays = list(break_array_into_chunks(speech_array, int(SR * frame_duration)))
    return arrays


# fmt: off
expected_speech_probas=[0.124, 0.035, 0.644, 0.324, 0.455, 0.111, 0.648, 0.662, 0.954, 0.985, 0.963, 0.681, 0.293, 0.635, 0.99, 0.975, 1.0, 0.987, 0.978, 0.941, 1.0, 1.0, 1.0, 0.999, 0.999, 0.987, 0.989, 0.998, 0.999, 0.996]
# fmt: on


def test_nemo_vad(vad: NeMoVAD, audio_arrays):
    probas = np.array([vad.predict(signal).probs_speech for signal in audio_arrays])
    assert_allclose(np.array(expected_speech_probas), probas, atol=0.01)


# @pytest.fixture()
# def nemo_segmentor(vad):
#     frame_len = 0.1
#     segmenter = StreamingSignalSegmentor(frame_len, vad=vad)
#     segmenter.init()
#     return segmenter
#
#
# def test_vad_segmentation(nemo_segmentor):
#     speech_arrays = list(tqdm(nemo_segmentor.speech_arrays_from_file(file)))
#     assert len(speech_arrays) > 0

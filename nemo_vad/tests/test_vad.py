import numpy as np
import pytest
from numpy.testing import assert_allclose
from tqdm import tqdm

from ml4audio.audio_utils.audio_io import (
    load_and_resample_16bit_PCM,
    break_array_into_chunks,
)
from nemo_vad.nemo_streaming_vad import NeMoVAD
from nemo_vad.streaming_vad_segmentation import StreamingSignalSegmentor

raw_audio_chunks_dur = 0.01

file = "nemo_vad/tests/resources/VAD_demo.wav"
SR = 16_000


@pytest.fixture()
def audio_arrays():
    speech_array = load_and_resample_16bit_PCM(file, SR)
    arrays = list(break_array_into_chunks(speech_array, int(SR * raw_audio_chunks_dur)))
    return arrays


# fmt: off
expected_speech_probas=[0.124, 0.035, 0.644, 0.324, 0.455, 0.111, 0.648, 0.662, 0.954, 0.985, 0.963, 0.681, 0.293, 0.635, 0.99, 0.975, 1.0, 0.987, 0.978, 0.941, 1.0, 1.0, 1.0, 0.999, 0.999, 0.987, 0.989, 0.998, 0.999, 0.996]
# fmt: on


def test_nemo_vad():
    speech_array = load_and_resample_16bit_PCM(file, SR)
    arrays = list(break_array_into_chunks(speech_array, int(SR * 0.1)))

    vad = NeMoVAD(
        threshold=0.3,
        frame_duration=0.1,
        window_len_in_secs=4 * 0.1,
        input_sample_rate=SR,
    ).build()
    probas = np.array([vad.predict(signal).probs_speech for signal in arrays])
    assert_allclose(np.array(expected_speech_probas), probas, atol=0.01)


@pytest.mark.parametrize(
    "params",
    [
        (0.3, 0.05, 42840, 3),
        (0.5, 0.1, 46040, 2),
        (0.8, 0.1, 36440, 3),
    ],
)
def test_vad_segmentation(params, audio_arrays):
    threshold, frame_duration, exp_voice_dur, num_segs = params
    # assert frame_duration>raw_audio_chunks_dur
    vad = NeMoVAD(
        threshold=threshold,
        frame_duration=frame_duration,
        window_len_in_secs=4 * frame_duration,
        input_sample_rate=SR,
    )
    segmenter = StreamingSignalSegmentor(vad=vad).build()

    g = (segmenter.handle_audio_array(a) for a in audio_arrays)
    voice_segs = [vs for vs in g if vs is not None and vs.is_final()]
    last_seg = segmenter.flush()
    if last_seg is not None:
        voice_segs = voice_segs + [last_seg]

    voice_dur = sum([len(vs.array) for vs in voice_segs])
    audio_dur = sum(len(a) for a in audio_arrays)
    assert audio_dur == 47640
    assert voice_dur == exp_voice_dur
    assert len(voice_segs) == num_segs

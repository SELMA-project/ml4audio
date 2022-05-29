import numpy as np
from numpy.testing import assert_allclose

from ml4audio.asr_inference.logits_cutter import LogitsCutter
from ml4audio.audio_utils.overlap_array_chunker import (
    MessageChunk,
)
from conftest import overlapping_messages_from_array


def test_logits_cutter(
    librispeech_logtis_file,
):

    logits = np.load(librispeech_logtis_file, allow_pickle=True).squeeze()
    logits_chunks: list[MessageChunk] = list(
        overlapping_messages_from_array(logits, step_size=100, chunk_size=200)
    )
    chunk_spans = [
        (ch.array, (ch.frame_idx, ch.frame_idx + len(ch.array))) for ch in logits_chunks
    ]

    lc = LogitsCutter()

    left_right = [lc.calc_left_right(l, s_e) for l, s_e in chunk_spans]
    print(f"{[(l.shape if l is not None else 0,r.shape) for l,r in left_right ]=}")
    parts = [l for l, r in left_right] + [left_right[-1][1]]
    parts = [x for x in parts if x is not None]

    assert_allclose(np.concatenate(parts), logits)

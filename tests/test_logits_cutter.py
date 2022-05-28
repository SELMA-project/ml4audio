import numpy as np
from beartype import beartype
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from ml4audio.asr_inference.logits_cutter import LogitsCutter
from ml4audio.audio_utils.audio_io import break_array_into_chunks
from ml4audio.audio_utils.overlap_array_chunker import (
    OverlapArrayChunker,
    messages_from_chunks,
    MessageChunk,
)


@beartype
def overlapping_messages_from_array(
    audio_array: NDArray, step_size: int, chunk_size: int
):
    chunker = OverlapArrayChunker(
        chunk_size=chunk_size,
        min_step_size=step_size,
    )
    chunker.reset()

    small_chunks = break_array_into_chunks(audio_array, chunk_size=100)
    chunks_g = (
        am
        for ch in messages_from_chunks("dummy-id", small_chunks)
        for am in chunker.handle_datum(ch)
    )
    yield from chunks_g


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

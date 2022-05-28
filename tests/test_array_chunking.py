from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import pytest
from beartype import beartype

from ml4audio.audio_utils.audio_message_chunking import AudioMessageChunk, \
    _DONT_EMIT_PREMATURE_CHUNKS, DONT_EMIT_PREMATURE_CHUNKS, OverlappingArrayChunker


@beartype
def chunk_test_data(seq: Iterable, chunk_len: int):
    buffer = []
    for k in seq:
        if len(buffer) >= chunk_len:
            yield buffer[:chunk_len]
            buffer = buffer[chunk_len:]
        buffer.append(k)

    if len(buffer) > 0:
        yield buffer


TestSequence = list[list[int]]


@beartype
def build_test_chunks(input_data: Iterable[TestSequence]) -> list[AudioMessageChunk]:
    def gen_seq(test_chunks):
        frame_idx = np.cumsum([0] + [len(tc) for tc in test_chunks[:-1]]).tolist()
        yield from [
            AudioMessageChunk(
                message_id=f"test-message",
                frame_idx=k,
                array=np.array(chunk, dtype=np.int16),
                end_of_signal=k == frame_idx[-1],
            )
            for k, chunk in zip(frame_idx, test_chunks)
        ]

    input_chunks = [x for tc in input_data for x in gen_seq(tc)]
    return input_chunks


@dataclass
class TestCase:
    input_chunks: list[AudioMessageChunk]
    chunk_size: int
    step_size: Optional[int]
    expected: list[int]
    minimum_chunk_size: Union[int, _DONT_EMIT_PREMATURE_CHUNKS]
    max_step_size: Optional[int] = None


test_case_0 = TestCase(
    input_chunks=build_test_chunks(
        [
            list(chunk_test_data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2)),
        ]
    ),
    chunk_size=4,
    step_size=2,
    expected=[0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9],
    # + [8, 9],  # cropped,
    minimum_chunk_size=DONT_EMIT_PREMATURE_CHUNKS,
)
premature_chunk = [0, 1]

test_case_premature = TestCase(
    build_test_chunks(
        [
            list(chunk_test_data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2)),
        ]
    ),
    4,
    2,
    premature_chunk + [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9],
    # + [8, 9],  # cropped,
    2,
)
premature_chunk_1 = [0, 1]
premature_chunk_2 = [0, 1, 2, 3]

test_case_premature_1 = TestCase(
    build_test_chunks(
        [
            list(chunk_test_data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)),
        ]
    ),
    6,  # chunk_size
    2,  # step_size
    premature_chunk_1
    + premature_chunk_2
    + [0, 1, 2, 3, 4, 5]
    + [2, 3, 4, 5, 6, 7]
    + [4, 5, 6, 7, 8, 9],
    # + [ 6, 7, 8, 9], # do not yield this cropped
    2,
)
test_case_premature_2 = TestCase(
    build_test_chunks(
        [
            list(chunk_test_data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2)),
        ]
    ),
    6,  # chunk_size
    2,  # step_size
    premature_chunk_1
    + premature_chunk_2
    + [0, 1, 2, 3, 4, 5]
    + [2, 3, 4, 5, 6, 7]
    + [4, 5, 6, 7, 8, 9],
    # + [ 6, 7, 8, 9], # do not yield this cropped
    2,
)

premature_2_flush_no_cropped = TestCase(
    build_test_chunks(
        [
            list(chunk_test_data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)),
        ]
    ),
    6,  # chunk_size
    2,  # step_size
    premature_chunk_1
    + premature_chunk_2
    + [0, 1, 2, 3, 4, 5]
    + [2, 3, 4, 5, 6, 7]
    + [4, 5, 6, 7, 8, 9]
    + [
        5,
        6,
        7,
        8,
        9,
        10,
    ],  # flushed ending, here stepped by one! even though step-size is not variable!
    2,
)

test_case_premature_3_no_cropped = TestCase(
    build_test_chunks(
        [
            list(chunk_test_data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)),
        ]
    ),
    6,  # chunk_size
    2,  # step_size
    premature_chunk_1
    + premature_chunk_2
    + [0, 1, 2, 3, 4, 5]
    + [2, 3, 4, 5, 6, 7]
    + [4, 5, 6, 7, 8, 9],
    # + [6, 7, 8, 9],  # no cropped!
    2,
)

test_case_premature_3_varlen = TestCase(
    build_test_chunks(
        [
            [[0], [1], [2, 3], [4, 5], [6, 7, 8, 9], [10, 11]],
        ]
    ),
    chunk_size=6,  # chunk_size
    step_size=1,  # step_size
    expected=[0]  # premature_chunk_0
    + premature_chunk_1
    + premature_chunk_2
    + [0, 1, 2, 3, 4, 5]
    + [3, 4, 5, 6, 7, 8]  # stepped 3
    + [4, 5, 6, 7, 8, 9]  # stepped 1
    + [6, 7, 8, 9, 10, 11],  # stepped 2
    minimum_chunk_size=1,
    max_step_size=3,
)


# @pytest.mark.parametrize(
#     "test_case",
#     [
#         test_case_0,
#         test_case_premature,
#         test_case_premature_2,
#     ],
# )
# def test_audio_message_chunker_generator(test_case: TestCase):
#
#     ab = AudioMessageChunker(
#         test_case.chunk_size,
#         test_case.step_size,
#         minimum_chunk_size=test_case.minimum_chunk_size,
#     )
#     out = [
#         x.audio_array
#         for x in audio_message_chunker_generator(ab, test_case.input_chunks)
#     ]
#
#     pred = np.concatenate(out).tolist()
#     assert pred == test_case.expected


@pytest.mark.parametrize(
    "test_case",
    [
        test_case_0,
        test_case_premature,
        test_case_premature_1,
        test_case_premature_2,
        premature_2_flush_no_cropped,
        test_case_premature_3_no_cropped,
        test_case_premature_3_varlen,
    ],
)
def test_AudioMessageChunker(test_case: TestCase):

    ab = OverlappingArrayChunker(
        test_case.chunk_size,
        test_case.step_size,
        minimum_chunk_size=test_case.minimum_chunk_size,
        max_step_size=test_case.max_step_size,
    )
    ab.reset()
    messages = [x for m in test_case.input_chunks for x in ab.handle_datum(m)]
    arrays = [m.array for m in messages]
    pred = [str(i) for i in np.concatenate(arrays).tolist()]
    expected = [str(i) for i in test_case.expected]
    assert pred == expected

from dataclasses import field, dataclass
from math import floor, ceil
from typing import Optional

from beartype import beartype

from misc_utils.beartypes import NumpyFloat2DArray


@dataclass
class LogitsCutter:
    """
    consider having some iterable of somehow overlapping chunks of logits
    this LogitsCutter takes a logits_chunk cuts away the overlaps (left & right) and
     returns a non-overlapping left-part and right-part (whos right end will be cut away "next time")
    """

    _buffer: NumpyFloat2DArray = field(init=False)
    _last_end: int = field(init=False, default=0)

    def reset(self):
        self._buffer = None
        self._last_end = 0

    @beartype
    def calc_left_right(
        self, logits_chunk: NumpyFloat2DArray, start_end: tuple[int, int]
    ) -> tuple[Optional[NumpyFloat2DArray], NumpyFloat2DArray]:
        """
        based on: https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/src/transformers/pipelines/automatic_speech_recognition.py#L337
        """
        a_start, a_end = start_end
        if self._last_end > 0:
            audio_slice_len = a_end - a_start
            logit_window_len = logits_chunk.shape[0]
            audio_to_logits_ratio = audio_slice_len / logit_window_len
            logits_overlap = (self._last_end - a_start) / audio_to_logits_ratio
            right_part = logits_chunk[floor(logits_overlap / 2) :]
            left_part = self._buffer[: -ceil(logits_overlap / 2)]

        else:
            right_part = logits_chunk
            left_part = None

        self._buffer = right_part
        self._last_end = a_end

        return left_part, right_part
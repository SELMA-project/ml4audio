import difflib
import os
from dataclasses import field, dataclass
from typing import Optional

import numpy as np
from beartype import beartype

from misc_utils.buildable import Buildable
from misc_utils.utils import just_try
from ml4audio.asr_inference.transcript_glueing import (
    calc_new_suffix,
    NO_NEW_SUFFIX,
)
from ml4audio.audio_utils.aligned_transcript import (
    TimestampedLetters,
)

DEBUG = os.environ.get("DEBUG", "False").lower() != "false"
if DEBUG:
    print("TranscriptGluer DEBUGGING MODE")


@dataclass
class ASRStreamInferenceOutput:
    id: str
    aligned_transcript: TimestampedLetters  # TODO: rename
    end_of_message: bool = False


@dataclass
class TranscriptGluer(Buildable):
    """
    ───▄▄▄
    ─▄▀░▄░▀▄
    ─█░█▄▀░█
    ─█░▀▄▄▀█▄█▄▀
    ▄▄█▄▄▄▄███▀

    """

    _hyp_buffer: Optional[TimestampedLetters] = field(
        init=False, repr=False, default=None
    )
    seqmatcher: Optional[difflib.SequenceMatcher] = field(
        init=False, repr=False, default=None
    )

    def __enter__(self):
        return self.build()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def reset(self) -> None:
        self._hyp_buffer: Optional[TimestampedLetters] = None

    def _build_self(self):
        self.reset()
        self.seqmatcher = difflib.SequenceMatcher()

    @beartype
    def calc_transcript_suffix(self, inp: TimestampedLetters) -> TimestampedLetters:

        if self._hyp_buffer is None:
            self._hyp_buffer = inp
            new_suffix = inp
        else:
            new_suffix = just_try(
                lambda: calc_new_suffix(
                    left=self._hyp_buffer, right=inp, sm=self.seqmatcher
                ),
                default=NO_NEW_SUFFIX,
                # a failed glue does not add anything! In the hope that overlap is big enough so that it can be recovered by next glue!
                verbose=DEBUG,
                print_stacktrace=False,
                reraise=True,
            )
            KEEP_DURATION = 100  # was not working with 40
            self._hyp_buffer = self._hyp_buffer.slice(
                np.argwhere(self._hyp_buffer.timestamps < new_suffix.timestamps[0])
            )
            self._hyp_buffer = TimestampedLetters(
                self._hyp_buffer.letters + new_suffix.letters,
                np.concatenate([self._hyp_buffer.timestamps, new_suffix.timestamps]),
            )
            self._hyp_buffer = self._hyp_buffer.slice(
                np.argwhere(
                    self._hyp_buffer.timestamps
                    > self._hyp_buffer.timestamps[-1] - KEEP_DURATION
                )
            )

        return new_suffix

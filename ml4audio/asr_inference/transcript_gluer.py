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

    _prefix: Optional[TimestampedLetters] = field(init=False, repr=False, default=None)
    seqmatcher: Optional[difflib.SequenceMatcher] = field(
        init=False, repr=False, default=None
    )

    def __enter__(self):
        return self.build()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def reset(self) -> None:
        self._prefix: Optional[TimestampedLetters] = None

    def _build_self(self):
        self.reset()
        self.seqmatcher = difflib.SequenceMatcher()

    @beartype
    def calc_transcript_suffix(self, inp: TimestampedLetters) -> TimestampedLetters:

        if self._prefix is None:
            self._prefix, new_suffix = inp, inp
        else:
            self._prefix, new_suffix = self._calc_prefix_suffix(self._prefix, inp)

        return new_suffix

    def _calc_prefix_suffix(self, prefix: TimestampedLetters, inp: TimestampedLetters):
        new_suffix = just_try(
            lambda: calc_new_suffix(left=prefix, right=inp, sm=self.seqmatcher),
            default=NO_NEW_SUFFIX,
            # a failed glue does not add anything! In the hope that overlap is big enough so that it can be recovered by next glue!
            verbose=DEBUG,
            print_stacktrace=False,
            reraise=False,
        )
        KEEP_DURATION = 100  # was not working with 40
        prefix = prefix.slice(np.argwhere(prefix.timestamps < new_suffix.timestamps[0]))
        prefix = TimestampedLetters(
            prefix.letters + new_suffix.letters,
            np.concatenate([prefix.timestamps, new_suffix.timestamps]),
        )
        prefix = prefix.slice(
            np.argwhere(prefix.timestamps > prefix.timestamps[-1] - KEEP_DURATION)
        )
        return prefix, new_suffix

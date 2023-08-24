import difflib
import os
from dataclasses import field, dataclass
from typing import Optional, Union

import numpy as np
from beartype import beartype

from misc_utils.buildable import Buildable
from misc_utils.utils import just_try
from ml4audio.asr_inference.transcript_glueing import (
    calc_new_suffix,
    NO_NEW_SUFFIX,
    _NO_NEW_SUFFIX,
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
    def calc_transcript_suffix(
        self, inp: TimestampedLetters
    ) -> Union[TimestampedLetters, _NO_NEW_SUFFIX]:

        if self._prefix is None:
            self._prefix, new_suffix = inp, inp
        else:
            self._prefix, new_suffix = self._calc_glued_and_suffix(self._prefix, inp)

        return new_suffix

    @beartype
    def _calc_glued_and_suffix(
        self, prefix: TimestampedLetters, inp: TimestampedLetters
    ) -> tuple[TimestampedLetters, Union[TimestampedLetters, _NO_NEW_SUFFIX]]:
        new_suffix = just_try(
            lambda: calc_new_suffix(left=prefix, right=inp, sm=self.seqmatcher),
            default=NO_NEW_SUFFIX,
            # a failed glue does not add anything! In the hope that overlap is big enough so that it can be recovered by next glue!
            verbose=DEBUG,
            print_stacktrace=True,
            reraise=False,
        )
        if new_suffix is not NO_NEW_SUFFIX:
            glued_trimmed = self._glue_and_trim(prefix, new_suffix)
        else:
            glued_trimmed = prefix
        return glued_trimmed, new_suffix

    def _glue_and_trim(self, prefix, new_suffix):
        KEEP_DURATION = 100  # was not working with 40
        prefix_to_keep = prefix.slice(
            np.argwhere(prefix.timestamps < new_suffix.timestamps[0])
        )
        glued = TimestampedLetters(
            prefix_to_keep.letters + new_suffix.letters,
            np.concatenate([prefix_to_keep.timestamps, new_suffix.timestamps]),
        )
        glued_trimmed = glued.slice(
            np.argwhere(glued.timestamps > glued.timestamps[-1] - KEEP_DURATION)
        )
        return glued_trimmed

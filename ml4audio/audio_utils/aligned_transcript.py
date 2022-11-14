"""
copypasted from: https://github.com/dertilo/speech-to-text
"""
from dataclasses import dataclass
from typing import List, Optional, Union, Iterable

from beartype import beartype
from misc_utils.beartypes import NeList
from misc_utils.dataclass_utils import UNDEFINED


@dataclass
class LetterIdx:
    letter: str
    r_idx: int  #TODO: rename to audio_frame_idx

def letter_to_words(letters: Iterable[LetterIdx]) -> Iterable[list[LetterIdx]]:
    loow: list[LetterIdx] = []  # letters of one word
    for l in letters:
        if l.letter == " ":
            assert len(loow) > 0
            yield loow
            loow = []
        else:
            loow.append(l)


@dataclass
class AlignedTranscript:
    """
    input-sequence aligned
    frame_id:  is index of first audio-frame of the audio-chunk that lead to this transcript
    offset: is index of very-first letter, (so that r_idx of very first letter is 0, and thereby those indizes of LetterIdx are relative)
    offset >= frame_id
    """

    letters: list[LetterIdx]
    sample_rate: int
    offset: int = 0 # TODO: rename to audio_frame_offset

    logits_score: Optional[float] = None  # TODO: who needs this?
    lm_score: Optional[float] = None
    frame_id: Optional[int] = None  # TODO: rename to audio_segment_frame_idx

    def __post_init__(self):
        self.update_offset()

    def set_abs_pos_in_time(self, frame_id: int):
        """
        absolute positioning in time
        """
        self.frame_id = frame_id
        self.update_offset(add_to_offset=frame_id)

    @property
    def text(self):
        return "".join([x.letter for x in self.letters])

    @property
    def array_idx(self):
        return [x.r_idx for x in self.letters]

    def abs_idx(self, letter: LetterIdx):
        return self.offset + letter.r_idx

    def abs_timestamp(self, letter: LetterIdx):
        return self.abs_idx(letter) / self.sample_rate

    @property
    def rel_timestamps(self):
        return [l.r_idx / self.sample_rate for l in self.letters]

    @property
    def abs_timestamps(self) -> list[float]:
        return [(l.r_idx + self.offset) / self.sample_rate for l in self.letters]

    def __len__(self):
        return len(self.letters)

    @property
    def len_not_space(self):
        return len(self.text.replace(" ", ""))

    def update_offset(self, add_to_offset: int = 0):
        if len(self.letters) > 0:
            offset = self.letters[0].r_idx
            self.letters = [LetterIdx(l.letter, l.r_idx - offset) for l in self.letters]
            self.offset += offset + add_to_offset
        else:
            self.offset += add_to_offset  # should not matter

        return self

    @beartype
    def slice_subsegment(self, abs_start: int, abs_end: int) -> "AlignedTranscript":
        letters = [
            x
            for x in self.letters
            if self.abs_idx(x) >= abs_start and self.abs_idx(x) < abs_end
        ]
        return AlignedTranscript(
            letters=letters,
            sample_rate=self.sample_rate,
            logits_score=self.logits_score,
            lm_score=self.lm_score,
            frame_id=self.frame_id,
        ).update_offset(self.offset)

    @beartype
    def slice_via_timestamps(self, start: float, end: float) -> "AlignedTranscript":
        letters = [
            l
            for l in self.letters
            if self.abs_timestamp(l) >= start and self.abs_timestamp(l) < end
        ]
        return AlignedTranscript(
            letters=letters,
            sample_rate=self.sample_rate,
            logits_score=self.logits_score,
            lm_score=self.lm_score,
            frame_id=self.frame_id,
        ).update_offset(self.offset)

    def remove_unnecessary_spaces(self):
        """
        unnecessary: spaces at start/end, consecutive spaces
        """
        first_letter = [self.letters[0]] if self.letters[0].letter != " " else []
        last_letter = [self.letters[-1]] if self.letters[-1].letter != " " else []
        self.letters = (
            first_letter
            + [
                self.letters[k]
                for k in range(1, len(self.letters) - 1)
                if self.letters[k].letter != " " or self.letters[k - 1].letter != " "
            ]
            + last_letter
        )
        assert not any(
            (
                self.text[k] == " " and self.text[k + 1] == " "
                for k in range(len(self.text) - 1)
            )
        )


@dataclass
class NeAlignedTranscript(AlignedTranscript):
    # TODO(tilo): WTF see NonEmptyAlignedTranscript !!!
    letters: NeList[LetterIdx] = UNDEFINED
    sample_rate: int = UNDEFINED

    def slice_subsegment(self, abs_start: int, abs_end: int) -> "NeAlignedTranscript":
        """
        TODO: anti-pattern here! this insertion of a dummy space letter (" ")
            is very use-case/situation specific, should not be done in a class!
            -> too much coupling! this class is spanning to many contexts!
        """
        letters = [
            x
            for x in self.letters
            if self.abs_idx(x) >= abs_start and self.abs_idx(x) < abs_end
        ]
        if len(letters) == 0:
            print(
                f"{self.__class__.__name__}: slice_subsegment made me empty! creating dummy letter"
            )
            letters = [LetterIdx(" ", 0)]

        return NeAlignedTranscript(
            letters=letters,
            sample_rate=self.sample_rate,
            logits_score=self.logits_score,
            lm_score=self.lm_score,
            frame_id=self.frame_id,
        ).update_offset(self.offset)

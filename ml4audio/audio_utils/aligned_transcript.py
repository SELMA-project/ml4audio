"""
copypasted from: https://github.com/dertilo/speech-to-text
"""
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class LetterIdx:
    letter: str
    r_idx: int  # relative index in input sequence not towards logits!!


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
    offset: int = 0

    logits_score: Optional[float] = None  # TODO: who needs this?
    lm_score: Optional[float] = None
    frame_id: Optional[int] = None  # TODO: how is this different from offset?

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

from dataclasses import dataclass

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from misc_utils.beartypes import NeNpFloatDim1


@dataclass
class TimestampedLetters:
    letters: str
    timestamps: NeNpFloatDim1

    def __post_init__(self):
        self.validate_data()

    def validate_data(self):
        strictly_increasing = np.all(np.diff(self.timestamps) >= 0)
        assert (
            strictly_increasing
        ), f"{self.timestamps=}\n{np.argwhere(np.diff(self.timestamps)<=0)}"
        assert len(self.letters) == len(self.timestamps)

    def __len__(self):
        return len(self.letters)

    @beartype
    def slice(self, those: NDArray[int]):
        those = those.squeeze(1)
        sliced = TimestampedLetters(
            "".join([self.letters[i] for i in those]), self.timestamps[those]
        )
        return sliced

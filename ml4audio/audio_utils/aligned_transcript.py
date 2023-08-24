from dataclasses import dataclass

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from misc_utils.beartypes import NeNumpyFloat1DArray


@dataclass
class TimestampedLetters:
    letters: str
    timestamps: NeNumpyFloat1DArray

    def __post_init__(self):
        self.validate_data()

    def validate_data(self):
        strictly_increasing = np.all(np.diff(self.timestamps) > 0)
        assert strictly_increasing, f"{self.timestamps=}"
        assert len(self.letters) == len(self.timestamps)

    def __len__(self):
        return len(self.letters)

    @beartype
    def slice(self, those: NDArray[np.int]):
        those = those.squeeze()
        sliced = TimestampedLetters(
            "".join([self.letters[i] for i in those]), self.timestamps[those]
        )
        return sliced

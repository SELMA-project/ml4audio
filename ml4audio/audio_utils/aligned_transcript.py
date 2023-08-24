from dataclasses import dataclass
from typing import List, Optional, Union, Iterable

import numpy as np
from beartype import beartype
from numpy._typing import NDArray

from misc_utils.beartypes import NeList, NeNumpyFloat1DArray
from misc_utils.dataclass_utils import UNDEFINED


@dataclass
class TimestampedLetters:
    letters: str
    timestamps: NeNumpyFloat1DArray

    def __post_init__(self):
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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class LangClf:
    @abstractmethod
    def init(self):
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def predict(self, audio_array: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError

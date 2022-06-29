import itertools
from dataclasses import dataclass
from typing import Optional, Iterator, Union

import math
import torch
from beartype import beartype

from huggingface_wav2vec2_finetuning.data_loading_commons import IterableDatasetBase
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.utils import buffer_shuffle

from ml4audio.audio_utils.audio_data_models import AudioTextData, ArrayText


@beartype
def calc_this_workers_start_end(start: int, end: int) -> tuple[int, int]:
    """
    see: https://github.com/pytorch/pytorch/blob/f2582a59d0835323ebf143726ea79ba52e7cceff/torch/utils/data/dataset.py#L128

    TODO: actually this is a stupid idea! would be better if kth-worker would "not-skip" every k-th sample
        thereby no need to eath large portions of the entire input-iterable! which can be very expensive!
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
        iter_start = start
        iter_end = end
    else:  # in a worker process
        per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        iter_start = start + worker_id * per_worker
        iter_end = min(iter_start + per_worker, end)
        print(f"{worker_id=}: {iter_start=}, {iter_end=}")
    return iter_start, iter_end


@dataclass
class StartEndIterableDataset(IterableDatasetBase, Buildable):
    """
    multiple data-loaders reading from corpus need to start-end at different "points" in the iterable
    """

    array_texts: Union[_UNDEFINED, AudioTextData] = UNDEFINED
    limit: Optional[int] = None
    shufflebuffer_size: Optional[int] = None

    def __len__(self):
        return self.limit

    @beartype
    def _generate_array_texts(self) -> Iterator[ArrayText]:
        iter_start, iter_end = calc_this_workers_start_end(0, self.limit)
        g = (
            (a, t)
            # for corpus in self.corpus
            for a, t in self.array_texts
        )
        array_text_g = itertools.islice(g, iter_start, iter_end)
        if self.shufflebuffer_size is not None:
            g = buffer_shuffle(array_text_g, buffer_size=self.shufflebuffer_size)
        else:
            g = array_text_g
        return iter(g)

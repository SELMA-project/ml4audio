import itertools
from dataclasses import dataclass
from typing import Optional, List, Iterator, Union

import math
import numpy as np
import torch
from transformers import Wav2Vec2Processor

from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.utils import buffer_shuffle

from speech_data.asr_corpora import ArrayText, Auteda
from text_processing.asr_text_normalization import normalize_filter_text
from wav2vec2_finetuning.data_loading.data_loading_commons import IterableDatasetBase


def calc_this_workers_start_end(start, end):
    """
    see: https://github.com/pytorch/pytorch/blob/f2582a59d0835323ebf143726ea79ba52e7cceff/torch/utils/data/dataset.py#L128
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
class IterableASRCorporaDataset(IterableDatasetBase, Buildable):
    corpus: Union[_UNDEFINED, Auteda] = UNDEFINED
    limit: Optional[int] = None
    shufflebuffer_size: Optional[int] = None

    def __len__(self):
        return self.limit

    def _generate_array_texts(self) -> Iterator[ArrayText]:
        iter_start, iter_end = calc_this_workers_start_end(0, self.limit)
        g = (
            (a, t)
            # for corpus in self.corpus
            for a, t in self.corpus
        )
        array_text_g = itertools.islice(g, iter_start, iter_end)
        if self.shufflebuffer_size is not None:
            g = buffer_shuffle(array_text_g, buffer_size=self.shufflebuffer_size)
        else:
            g = array_text_g
        return iter(g)

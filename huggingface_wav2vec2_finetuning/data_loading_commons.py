import os
import sys
from abc import abstractmethod

from beartype import beartype
from nemo.collections.asr.parts.preprocessing import AudioAugmentor

from huggingface_wav2vec2_finetuning.base_model_for_finetuning import (
    BaseModelForFinetuning,
)
from misc_utils.beartypes import NumpyFloat1DArray
from huggingface_wav2vec2_finetuning.hf_finetune_utils import (
    apply_asr_processor,
    HfASRSample,
)


from typing import Union, Optional, Iterator, Iterable

from misc_utils.dataclass_utils import _UNDEFINED, UNDEFINED
from misc_utils.utils import just_try, TimedIterable

import logging

from dataclasses import dataclass, field, asdict

from ml4audio.audio_data.nemo_perturbation import (
    ProbaPerturbationDC,
    apply_nemo_perturbations_with_retry,
)
from ml4audio.audio_utils.audio_data_models import ArrayText

logging.getLogger("filelock._api").setLevel(logging.ERROR)
import torch
from transformers import set_seed


@dataclass
class IterableDatasetBase(torch.utils.data.IterableDataset):
    """"""

    # TODO: finetune_model really needed or just its processor?
    finetune_model: Union[_UNDEFINED, BaseModelForFinetuning] = UNDEFINED
    perturbations: Optional[list[ProbaPerturbationDC]] = None

    augmentor: Optional[AudioAugmentor] = field(default=None, init=False, repr=False)
    worker_idx: Optional[int] = field(default=None, init=False, repr=False)
    local_rank: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def __exit__(self):
        pass

    @abstractmethod
    def _generate_array_texts(self) -> Iterator[ArrayText]:
        raise NotImplementedError

    def _failsafe_feature_extraction(self, at_g: Iterable[ArrayText]):
        for array, text in at_g:
            datum = just_try(lambda: self.process_array_text(array, text), verbose=True)
            # datum = self.process_array_text(array, text)
            if datum is not None:
                yield datum
            else:
                print(f"{self.worker_idx}: got failed datum")

    def __iter__(self) -> Iterator[dict]:

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_idx = int(self.local_rank * 1000 + worker_info.id)
        else:
            worker_idx = 0

        set_seed(worker_idx)
        self.worker_idx = worker_idx

        array_texts = TimedIterable(
            self._generate_array_texts(),
            # weight_fun=lambda x: len(x), # len(x) should always be 2 for array-text tuple!
        )

        g = TimedIterable(
            self._failsafe_feature_extraction(array_texts),
        )
        for k, datum in enumerate(g):
            datum: HfASRSample
            yield asdict(datum)
            if k > 0 and k % 1000 == 0 or k == 10:
                consumer_stats = {
                    "data_consuming_speed": array_texts.speed,
                    "data_processing_speed": g.speed,
                    "overall_loading_speed": g.outcome
                    / (array_texts.duration + g.duration),
                    # "data_consuming_duration_in_sec": array_texts.duration,
                    # "data_processing_duration_in_sec": g.duration,
                }
                print(f"{worker_idx=},{self.local_rank=}: {consumer_stats=}")
                sys.stdout.flush()
                sys.stderr.flush()

    @beartype
    def process_array_text(self, array: NumpyFloat1DArray, text: str) -> HfASRSample:
        sr = self.finetune_model.processor.feature_extractor.sampling_rate
        assert sr == 16000
        array = apply_nemo_perturbations_with_retry(
            array, sample_rate=sr, augmentor=self.augmentor
        )
        text = self.finetune_model._transcript_normalizer.apply(text)
        assert text is not None
        datum = apply_asr_processor(array, text, self.finetune_model.processor)
        return datum

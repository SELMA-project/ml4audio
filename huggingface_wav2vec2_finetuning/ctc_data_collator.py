import traceback
from dataclasses import dataclass
from pprint import pprint
from typing import Union, Optional, Any

import torch
from beartype import beartype
from transformers import Wav2Vec2Processor, BatchFeature

from misc_utils.beartypes import NeList


@dataclass
class DataCollatorCTCWithPadding:
    """
    based on: https://github.com/huggingface/transformers/blob/b9bb417324c0d9013c505dc39c016ab9ca0e23c8/examples/research_projects/wav2vec2/run_common_voice.py#L143

    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    some_batch: Optional[Any] = None

    @beartype
    def _process_pad(self, features: NeList) -> BatchFeature:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # with self.processor.as_target_processor(): # tilo does not like this implicit processor switching
        labels_batch = self.processor.tokenizer.pad(  # explicitly use tokenizier here
            label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

    @beartype
    def __call__(
        self, features: NeList[dict[str, Union[list[int], torch.Tensor]]]
    ) -> BatchFeature:
        """
        TODO(tilo): why did I want a try-except here?
        """

        try:
            batch = self._process_pad(features)
            if self.some_batch is None:
                self.some_batch = batch

        except Exception as e:
            print(e)
            traceback.print_exc()
            print("WARNING: Collator failed!!!")
            pprint(f"input: {features}")
            batch = self.some_batch

        return batch

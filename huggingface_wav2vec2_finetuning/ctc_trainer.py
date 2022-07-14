from typing import Dict, Union, Any

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import Trainer

from huggingface_wav2vec2_finetuning.ctc_data_collator import DataCollatorCTCWithPadding
from huggingface_wav2vec2_finetuning.hf_finetune_utils import (
    ReduceLROnPlateauWithWarmup,
)


def dummpy_step(**kwargs):
    pass


class CTCTrainer(Trainer):
    """
    in dryrun-mode does only one single forward-pass
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        try:
            if (
                isinstance(self.lr_scheduler, ReduceLROnPlateauWithWarmup)
                and self.state.global_step % self.args.eval_steps == 0
            ):
                self.lr_scheduler.step(metrics=self.state.best_metric)
            loss_d = super().training_step(model, inputs)
        except Exception as e:
            err = "CUDA out of memory" if "CUDA out of memory" in str(e) else e
            print(f"train-step failed with: {err}")
            model.zero_grad()
            loss_d = torch.tensor(torch.nan)
        return loss_d

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        assert isinstance(
            self.data_collator, DataCollatorCTCWithPadding
        ), f"{type(self.data_collator)=}"

        """
        # https://pytorch.org/docs/stable/data.html

        # loading from an iterable-style dataset is roughly equivalent with:

        for data in iter(dataset):
            yield collate_fn(data)

        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

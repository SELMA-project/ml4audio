from typing import Dict, Union, Any, Optional

import datasets
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from transformers import Trainer, is_datasets_available
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
    DistributedLengthGroupedSampler,
)

from huggingface_wav2vec2_finetuning.ctc_data_collator import DataCollatorCTCWithPadding


def dummpy_step(**kwargs):
    pass


class CTCTrainer(Trainer):
    """
    in dryrun-mode does only one single forward-pass
    """

    def __init__(self, **kwargs):
        self.dryrun = kwargs.pop("dryrun", False)
        self.dummy_loss = None
        super().__init__(**kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        if self.dryrun:
            """
            even monkeypatching these step methods did not lead to expected behavior
            cannot really get pure dryrun cause transformers train.py does processing that takes computing time
            """
            self.optimizer.step = dummpy_step
            self.lr_scheduler.step = dummpy_step

        if not self.dryrun or self.dummy_loss is None:
            model.train()
            inputs = self._prepare_inputs(inputs)
            try:
                loss_d = self.__train_step(inputs, model)
                self.dummy_loss = loss_d
            except Exception as e:
                err = "CUDA out of memory" if "CUDA out of memory" in str(e) else e
                print(f"train-step failed with: {err}")
                model.zero_grad()
                loss_d = torch.tensor(torch.nan)
            return loss_d
        else:
            return self.dummy_loss

    def __train_step(self, inputs, model):
        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(
                    f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']"
                )
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()
        loss_d = loss.detach()
        return loss_d

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        """
        TODO!
        """
        if self.args.group_by_length:
            lengths = self.train_dataset.precomputed_lengths
            model_input_name = (
                self.tokenizer.model_input_names[0]
                if self.tokenizer is not None
                else None
            )
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )

        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        assert isinstance(self.data_collator, DataCollatorCTCWithPadding), f"{type(self.data_collator)=}"
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )

        if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
            # tilo: I do NOT want sharding!! cause this is pulling from iterable-dataset and discarding every k sample!
            # if self.args.world_size > 1:
            #     train_dataset = IterableDatasetShard(
            #         train_dataset,
            #         batch_size=self.args.train_batch_size,
            #         drop_last=self.args.dataloader_drop_last,
            #         num_processes=self.args.world_size,
            #         process_index=self.args.process_index,
            #     )
            # TODO(tilo): docs say: "When both batch_size and batch_sampler are None (default value for batch_sampler is already None), automatic batching is disabled."
            """
            # https://pytorch.org/docs/stable/data.html

            # loading from an iterable-style dataset is roughly equivalent with:

            for data in iter(dataset):
                yield collate_fn(data)

            """
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

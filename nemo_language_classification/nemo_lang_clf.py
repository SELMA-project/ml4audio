import os

import sys
from beartype import beartype

from misc_utils.beartypes import TorchTensorFloat2D, TorchTensorInt

from dataclasses import dataclass
from typing import Dict

import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils import logging
import numpy as np
from tqdm import tqdm

from ml4audio.audio_utils.audio_io import MAX_16_BIT_PCM, read_audio_chunks_from_file
from nemo_language_classification.common import LangClf


@dataclass
class NemoLangClf(LangClf):
    model_file: str

    def init(self):
        self.model = load_EncDecSpeakerLabelModel(self.model_file)
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_array: np.ndarray) -> Dict[str, float]:
        labels = list(self.model.cfg.train_ds.labels)
        sig, sig_len = preprocess_signal(audio_array)
        logits, _ = self.model.forward(input_signal=sig, input_signal_length=sig_len)

        probs = torch.softmax(logits, dim=-1).squeeze()
        label2proba = {k: float(p.numpy()) for k, p in zip(labels, probs)}
        # class_idx = np.argmax(probs)
        # class_label = labels[class_idx]
        return label2proba


@beartype
def load_EncDecSpeakerLabelModel(pretrained_model: str) -> EncDecSpeakerLabelModel:
    """
    based on: https://github.com/NVIDIA/NeMo/blob/ddd87197e94ca23ae54e641dc7784e64c00a43d6/examples/speaker_tasks/recognition/speaker_reco_finetune.py#L63
    """
    if pretrained_model.endswith(".nemo"):
        logging.info(f"Using local speaker model from {pretrained_model}")
        model = EncDecSpeakerLabelModel.restore_from(restore_path=pretrained_model)
    elif pretrained_model.endswith(".ckpt"):
        logging.info(f"Using local speaker model from checkpoint {pretrained_model}")
        model = EncDecSpeakerLabelModel.load_from_checkpoint(
            checkpoint_path=pretrained_model
        )
    else:
        logging.info("Using pretrained speaker recognition model from NGC")
        model = EncDecSpeakerLabelModel.from_pretrained(model_name=pretrained_model)
    return model


def preprocess_signal(
    signal: np.ndarray,
) -> tuple[TorchTensorFloat2D, TorchTensorInt]:
    signal = signal.squeeze()
    assert signal.dtype == np.int16
    signal = signal.astype(np.float32) / MAX_16_BIT_PCM
    return (
        torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor([signal.size], dtype=torch.int64),
    )


if __name__ == "__main__":
    mdl = NemoLangClf(
        model_file=f"{os.environ['BASE_PATH']}/results/TRAINING/LANG_CLF/debug/SpeakerNet/2021-07-23_10-14-04/checkpoints/SpeakerNet--val_loss=6.84-epoch=1-last.ckpt"
    )
    mdl.init()
    input_sample_rate = 16000
    frame_duration = 4.0

    wav_file = "tests/resources/tuda_2015-02-03-13-51-36_Realtek.wav"
    for chunk in tqdm(
        read_audio_chunks_from_file(
            wav_file, input_sample_rate, chunk_duration=frame_duration
        )
    ):
        print(mdl.predict(chunk))

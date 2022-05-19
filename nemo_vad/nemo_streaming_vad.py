import copy
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
import torch
from beartype import beartype
from nemo.collections import asr as nemo_asr
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from numpy.typing import NDArray
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm


"""
based on: https://github.com/NVIDIA/NeMo/blob/v1.0.0/tutorials/asr/07_Online_Offline_Microphone_VAD_Demo.ipynb

"""
TARGET_SAMPLE_RATE = 16_000
DEBUG = False

if DEBUG:
    debug_name = "16kHz"
    # debug_name= "8kHz"
    DEBUG_DIR = f"{BASE_PATHES['base_path']}/data/tmp/debug_vad_{debug_name}"
    shutil.rmtree(DEBUG_DIR, ignore_errors=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)


def infer_signal(model: EncDecClassificationModel, batch):
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = (
        audio_signal.to(model.device),
        audio_signal_len.to(model.device),
    )
    logits = model.forward(
        input_signal=audio_signal, input_signal_length=audio_signal_len
    )
    return logits


counter = 0


@dataclass
class VADOutput:
    label_id: int
    label: str
    probs_background: float
    probs_speech: float
    logits: str


@dataclass
class NeMoVAD(Buildable):
    """
    TODO: this thing has a buffer -> state might be better treating it as StreamProcessor
        like intended months ago

    """

    vad_model_name: str = "vad_marblenet"
    threshold: float = 0.5
    frame_duration: float = 0.1
    window_len_in_secs: float = 0.5  # seconds
    input_sample_rate: Optional[int] = None

    buffer: NDArray = field(init=False, repr=False)
    buffer_size: int = field(init=False, repr=False)
    frame_len: int = field(init=False, repr=False)
    vocab: list[str] = field(init=False, repr=False)

    # # TODO: does this really make any sense? why streaming here?
    # def process(self, input_it: Iterator[np.ndarray]) -> Iterator[bool]:
    #     for array in input_it:
    #         yield self.is_speech(array)

    def _build_self(self) -> "NeMoVAD":
        self.cfg, self.vad_model = load_vad_model(self.vad_model_name)
        self.vocab = list(self.cfg.labels)
        self.vocab.append("_")

        sample_rate = self.cfg.train_ds.sample_rate
        assert sample_rate == TARGET_SAMPLE_RATE
        self.frame_len = int(self.frame_duration * self.input_sample_rate)

        self.buffer_size = int(self.window_len_in_secs * self.input_sample_rate)
        self.reset()
        return self

    def is_speech(self, frame: np.ndarray) -> bool:
        o = self.predict(frame)
        return True if o.label_id == 1 else False

    def build_batch(
        self,
        signal: np.ndarray,
    ):
        """
        TODO: who needs batch?
            this EncDecClassificationModel always expects batches?
        """
        assert signal.dtype == np.int16
        signal = signal.astype(np.float32) / MAX_16_BIT_PCM
        tgt_sr = self.cfg.train_ds.sample_rate
        if self.input_sample_rate != tgt_sr:
            signal = librosa.resample(
                signal,
                self.input_sample_rate,
                tgt_sr,
                scale=True,
                res_type="kaiser_best",
            )

        return (
            torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor([signal.size], dtype=torch.int64),
        )

    @beartype
    @torch.no_grad()
    def predict(self, frame: NDArray) -> VADOutput:
        """
        buffering logic see: https://github.com/NVIDIA/NeMo/blob/v1.0.0/tutorials/asr/07_Online_Offline_Microphone_VAD_Demo.ipynb
        """

        if len(frame) < self.frame_len:
            # print(
            #     "WARNING: this should happend only at end of a stream! i.e. when coming from file"
            # )
            frame = np.pad(frame, [0, self.frame_len - len(frame)], "constant")
        elif len(frame) > self.frame_len:
            assert False

        assert (
            len(frame) == self.frame_len
        ), f"len(frame)={len(frame)}, self.n_frame_len={self.frame_len}"
        self.buffer[: -self.frame_len] = self.buffer[self.frame_len :]
        self.buffer[-self.frame_len :] = frame
        logits = (
            infer_signal(self.vad_model, self.build_batch(self.buffer)).cpu().numpy()[0]
        )

        assert logits.shape[0]
        probs = torch.softmax(torch.as_tensor(logits), dim=-1)
        probas_s = probs[1].item()
        label_id = 1 if probas_s >= self.threshold else 0
        return VADOutput(
            label_id,
            str(self.vocab[label_id]),
            probs[0].item(),
            probs[1].item(),
            str(logits),
        )

    def reset(self) -> None:
        """
        TODO: who need this?
        """
        self.buffer = np.zeros(shape=self.buffer_size, dtype=np.int16)


@beartype
def load_vad_model(
    model_name: str = "vad_marblenet",
) -> tuple[DictConfig, EncDecClassificationModel]:
    vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name)
    cfg = copy.deepcopy(vad_model._cfg)
    vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
    vad_model.eval()
    vad_model = vad_model.to(vad_model.device)
    return cfg, vad_model


if __name__ == "__main__":

    vad = NeMoVAD(
        vad_model_name="vad_marblenet",
        threshold=0.3,
        frame_duration=0.1,
        window_len_in_secs=0.5,
        input_sample_rate=16000,
    ).build()

    wav_file = f"tests/resources/VAD_demo.wav"
    for chunk in tqdm(
        resample_stream_file(
            wav_file, vad.input_sample_rate, chunk_duration=vad.frame_duration
        )
    ):
        vad.is_speech(chunk)

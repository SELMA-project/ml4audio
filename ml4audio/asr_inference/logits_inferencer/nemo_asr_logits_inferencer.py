import os
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from beartype import beartype

from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    ResamplingASRLogitsInferencer,
)
from misc_utils.beartypes import (
    NumpyFloat1DArray,
    NumpyFloat2DArray,
    NeList,
)
from ml4audio.audio_utils.audio_io import MAX_16_BIT_PCM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import soundfile as sf


@dataclass
class NemoASRLogitsInferencer(ResamplingASRLogitsInferencer):
    def _build_self(self):
        # see: tools/ctc_segmentation/scripts/run_ctc_segmentation.py in nemo-code
        model_name = self.checkpoint.model_name_or_path
        if os.path.exists(model_name):
            self._model = nemo_asr.models.EncDecCTCModel.restore_from(model_name)
        elif model_name in nemo_asr.models.EncDecCTCModel.get_available_model_names():
            self._model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                model_name, strict=False
            )
        else:
            raise ValueError(
                f"{model_name} not a valid model name or path. Provide path to the pre-trained checkpoint "
                f"or choose from {nemo_asr.models.EncDecCTCModel.list_available_models()}"
            )
        self._model.eval()
        self.move_to_device(DEVICE)
        return self

    @property
    @beartype
    def vocab(self) -> NeList[str]:
        vocabulary = self._model.cfg.decoder.vocabulary
        # see: tools/ctc_segmentation/scripts/run_ctc_segmentation.py in nemo-code
        vocabulary = ["ε"] + list(vocabulary)
        return vocabulary

    @beartype
    def calc_logsoftmaxed_logits(self, audio: NumpyFloat1DArray) -> NumpyFloat2DArray:
        device = next(self._model.parameters()).device
        audio_signal = torch.as_tensor(audio.reshape(1, -1), dtype=torch.float32)
        audio_signal_len = torch.as_tensor([audio.size], dtype=torch.int64)

        with torch.no_grad():
            log_probs, encoded_len, greedy_predictions = self._model(
                input_signal=audio_signal.to(device),
                input_signal_length=audio_signal_len.to(device),
            )
            log_probs = log_probs.cpu().squeeze()

        log_probs = self._post_process_for_ctc_alignment(log_probs)
        assert log_probs.shape[1] == len(
            self.vocab
        ), f"{log_probs.shape=},{len(self.vocab)}"
        return log_probs

    @beartype
    def _post_process_for_ctc_alignment(
        self, log_probs: NumpyFloat2DArray
    ) -> NumpyFloat2DArray:
        """
        see:nvidia-nemo-code:  tools/ctc_segmentation/scripts/run_ctc_segmentation.py
        """
        blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
        log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
        return log_probs

    def _write_audio(self, file: str, audio: NumpyFloat1DArray) -> str:
        """
        TODO: nemo only accepts audio-files as input
        """
        audio = audio / np.max(np.abs(audio)) * (MAX_16_BIT_PCM - 1)
        audio = audio.astype(np.int16)
        sf.write(file, audio, samplerate=16000)
        return file

    @beartype
    def batched_calc_logsoftmaxed_logits(
        self, audio: list[NumpyFloat1DArray]
    ) -> list[NumpyFloat2DArray]:
        # device = next(self.model.parameters()).device

        with torch.no_grad():
            with TemporaryDirectory(prefix="/tmp/nemo_tmp_dir") as tmpdir:
                audio_files = [
                    self._write_audio(f"{tmpdir}/{k}.wav", a)
                    for k, a in enumerate(audio)
                ]

                log_probs: list[np.ndarray] = self._model.transcribe(
                    paths2audio_files=audio_files, batch_size=10, logprobs=True
                )

        return [self._post_process_for_ctc_alignment(lp) for lp in log_probs]

    def calc_logits(self, audio) -> torch.Tensor:
        # TODO!
        raise NotImplementedError("TODO!")

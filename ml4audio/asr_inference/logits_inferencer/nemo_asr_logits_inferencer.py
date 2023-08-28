import os
from dataclasses import dataclass, field

import nemo.collections.asr as nemo_asr
import torch
from beartype import beartype
from nemo.collections.asr.models import EncDecCTCModel

from misc_utils.beartypes import (
    NumpyFloat1DArray,
    NeList,
    NeStr,
    TorchTensor2D,
)
from misc_utils.dataclass_utils import UNDEFINED
from misc_utils.utils import slugify_with_underscores
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    ASRLogitsInferencer,
)
from ml4audio.text_processing.asr_text_cleaning import Letters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class NemoASRLogitsInferencer(ASRLogitsInferencer):
    model_name_or_path: str = UNDEFINED
    _model: EncDecCTCModel = field(init=False)

    @property
    def name(self) -> NeStr:
        return slugify_with_underscores(self.model_name_or_path)

    def _build_self(self):
        # see: tools/ctc_segmentation/scripts/run_ctc_segmentation.py in nemo-code
        model_name = self.model_name_or_path
        if os.path.exists(model_name):
            raise NotImplementedError
            self._model = nemo_asr.models.EncDecCTCModel.restore_from(model_name)
        else:
            self._model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name, strict=False
            )
        # else:
        #     raise ValueError(
        #         f"{model_name} not a valid model name or path. Provide path to the pre-trained checkpoint "
        #         f"or choose from {nemo_asr.models.EncDecCTCModelBPE.list_available_models()}"
        #     )
        self._model.eval()
        self._model = self._model.to(DEVICE)
        return self

    @property
    @beartype
    def vocab(self) -> NeList[str]:
        vocabulary = self._model.cfg.decoder.vocabulary
        vocabulary = list(vocabulary)
        return vocabulary

    @property
    @beartype
    def letter_vocab(self) -> Letters:
        bad_letters = ["<", ">", "▁"]
        return [l for l in dict.fromkeys("".join(self.vocab)) if l not in bad_letters]

    @beartype
    def calc_logits(self, audio: NumpyFloat1DArray) -> TorchTensor2D:
        device = next(self._model.parameters()).device
        audio_signal = torch.as_tensor(audio.reshape(1, -1), dtype=torch.float32)
        audio_signal_len = torch.as_tensor([audio.size], dtype=torch.int64)

        with torch.no_grad():
            log_probs, _encoded_len, _greedy_predictions = self._model(
                input_signal=audio_signal.to(device),
                input_signal_length=audio_signal_len.to(device),
            )
            log_probs = log_probs.cpu().squeeze()

        return log_probs


# TODO: what about these?
#
#     @beartype
#     def calc_logsoftmaxed_logits(self, audio: NumpyFloat1DArray) -> NumpyFloat2DArray:
#         device = next(self._model.parameters()).device
#         audio_signal = torch.as_tensor(audio.reshape(1, -1), dtype=torch.float32)
#         audio_signal_len = torch.as_tensor([audio.size], dtype=torch.int64)
#
#         with torch.no_grad():
#             log_probs, encoded_len, greedy_predictions = self._model(
#                 input_signal=audio_signal.to(device),
#                 input_signal_length=audio_signal_len.to(device),
#             )
#             log_probs = log_probs.cpu().squeeze()
#
#         log_probs = self._post_process_for_ctc_alignment(log_probs)
#         assert log_probs.shape[1] == len(
#             self.vocab
#         ), f"{log_probs.shape=},{len(self.vocab)}"
#         return log_probs
#
#     @beartype
#     def _post_process_for_ctc_alignment(
#         self, log_probs: NumpyFloat2DArray
#     ) -> NumpyFloat2DArray:
#         """
#         see:nvidia-nemo-code:  tools/ctc_segmentation/scripts/run_ctc_segmentation.py
#         """
#         blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
#         log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
#         return log_probs

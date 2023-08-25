from dataclasses import field, dataclass
from functools import cached_property
from typing import Union, Optional

import torch
from beartype import beartype
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from misc_utils.beartypes import (
    NumpyFloat1DArray,
    TorchTensor2D,
    NeStr,
)
from misc_utils.dataclass_utils import UNDEFINED
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    ASRLogitsInferencer,
)
from ml4audio.asr_inference.logits_inferencer.huggingface_checkpoints import (
    HfModelFromCheckpoint,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HFWav2Vec2LogitsInferencer(ASRLogitsInferencer):

    checkpoint: Union[HfModelFromCheckpoint] = UNDEFINED
    do_normalize: bool = True  # TODO: not sure yet what is better at inference time

    _processor: Optional[Wav2Vec2Processor] = field(
        init=False, repr=False, default=None
    )
    _model: Optional[torch.nn.Module] = field(init=False, repr=False, default=None)

    @property
    @beartype
    def name(self) -> NeStr:
        return self.checkpoint.name  # cut_away_path_prefixes(self.model_name)

    def move_to_device(self, device):
        self._model = self._model.to(device)

    def _build_self(self) -> "HFWav2Vec2LogitsInferencer":
        self._processor = self._load_prepare_processor()
        self._model = Wav2Vec2ForCTC.from_pretrained(self.checkpoint.model_path)
        self.move_to_device(DEVICE)
        return self

    @beartype
    def _load_prepare_processor(self) -> Wav2Vec2Processor:
        pr = Wav2Vec2Processor.from_pretrained(self.checkpoint.model_path)
        pr.feature_extractor.do_normalize = self.do_normalize
        pr.feature_extractor.return_attention_mask = True
        return pr

    @cached_property
    def vocab(self) -> list[str]:
        return [
            k
            for k, i in sorted(
                self._processor.tokenizer.get_vocab().items(), key=lambda x: x[1]
            )
        ]

    @beartype
    def calc_logits(self, audio: NumpyFloat1DArray) -> TorchTensor2D:

        features = self._processor(
            audio,
            sampling_rate=self.asr_model_sample_rate,
            return_tensors="pt",
            # padding=True, #TODO why was this set to true?
            # return_attention_mask=True,
        )
        device = next(self._model.parameters()).device
        with torch.no_grad():
            logits = (
                self._model(
                    features.input_values.to(device),
                    attention_mask=features.attention_mask.to(device),
                )
                .logits.cpu()
                .squeeze()
            )
        assert logits.shape[1] == len(self.vocab), f"{logits.shape=},{len(self.vocab)=}"
        return logits


#
# @dataclass
# class OnnxHFWav2Vec2LogitsInferencer(HFWav2Vec2LogitsInferencer):
#     checkpoint: OnnxedHFCheckpoint = UNDEFINED
#
#     def _build_self(self) -> "OnnxHFWav2Vec2LogitsInferencer":
#         self._processor = self._load_prepare_processor()
#
#         import onnx
#
#         onnx_model = onnx.load(self.checkpoint.onnx_model)
#         onnx.checker.check_model(onnx_model)
#
#         import onnxruntime as rt
#
#         sess_options = rt.SessionOptions()
#         sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
#         self._session = rt.InferenceSession(self.checkpoint.onnx_model, sess_options)
#         return self
#
#     @beartype
#     def _infer_logits(self, features: BatchFeature) -> TorchTensor2D:
#         input_values = features.input_values
#         onnx_outputs = self._session.run(
#             None, {self._session.get_inputs()[0].name: input_values.numpy()}
#         )[0]
#         return torch.from_numpy(onnx_outputs.squeeze())

from dataclasses import field, dataclass
from typing import Union, List, Optional

import numpy as np
import torch
from beartype import beartype
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    ResamplingASRLogitsInferencer,
)
from misc_utils.beartypes import NumpyFloat2DArray, NumpyFloat1DArray, TorchTensor2D

HFWAV2VEC2_SAMPLE_RATE = 16_000  # TODO: somewhen this might be model dependent!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW_SPEECH = Union[
    np.ndarray, List[float], List[np.ndarray], List[List[float]]
]  # see: transformers/models/wav2vec2/feature_extraction_wav2vec2.py


@dataclass
class HFWav2Vec2LogitsInferencer(ResamplingASRLogitsInferencer):
    """
    TODO: input_sample_rate triggers different hash+cache
    """

    _processor: Optional[Wav2Vec2Processor] = field(
        init=False, repr=False, default=None
    )

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

    @property
    def vocab(self) -> list[str]:
        return list(self._processor.tokenizer.get_vocab().keys())

    @beartype
    def batched_calc_logsoftmaxed_logits(
        self, audio: list[NumpyFloat1DArray]
    ) -> list[NumpyFloat2DArray]:
        """
        TODO: just sequentially inference here -> no real batching!!
        """
        return [self.calc_logsoftmaxed_logits(a) for a in audio]

    @beartype
    def calc_logsoftmaxed_logits(self, audio: NumpyFloat1DArray) -> NumpyFloat2DArray:
        lpz = self.logsoftmax(self.calc_logits(audio))
        assert lpz.shape[1] == len(self.vocab), f"{lpz.shape=},{len(self.vocab)}"
        return lpz

    @beartype
    def calc_logits(self, audio: NumpyFloat1DArray) -> TorchTensor2D:
        # debug_dir = "debug_wav_files"
        # os.makedirs(debug_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        # soundfile.write(f"{debug_dir}/audio-{timestamp}.wav", audio, 16000)

        device = next(self._model.parameters()).device
        inputs = self._processor(
            audio,
            sampling_rate=HFWAV2VEC2_SAMPLE_RATE,
            return_tensors="pt",
            # padding=True, #TODO why was this set to true?
            # return_attention_mask=True,
        )
        with torch.no_grad():
            logits = (
                self._model(
                    inputs.input_values.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                )
                .logits.cpu()
                .squeeze()
            )
        assert logits.shape[1] == len(self.vocab), f"{logits.shape=},{len(self.vocab)=}"
        return logits

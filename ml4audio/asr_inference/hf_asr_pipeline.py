from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
from beartype import beartype
from transformers import (
    AutoTokenizer,
    pipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ProcessorWithLM,
    AutoFeatureExtractor,
)

from misc_utils.beartypes import NumpyInt16Dim1
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import UNDEFINED
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)
from ml4audio.text_processing.ctc_decoding import BaseCTCDecoder
from ml4audio.text_processing.pyctc_decoder import PyCTCKenLMDecoder
from pyctcdecode import BeamSearchDecoderCTC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@beartype
def prepare_decoder_and_feature_extractor(
    decoder: Union[str, BeamSearchDecoderCTC, None], model_id: str
) -> tuple[Optional[BeamSearchDecoderCTC], Wav2Vec2FeatureExtractor]:
    if decoder == "off-the-shelf":
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
        feature_extractor = processor.feature_extractor

        # # set language model attributes
        # params = {"alpha": 0.3, "beta": 0.1}
        # for attribute, value in params.items():
        #     processor._set_language_model_attribute(processor.decoder, attribute,
        #                                             value)
        decoder = processor.decoder

    elif isinstance(decoder, BeamSearchDecoderCTC):
        # processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
        # feature_extractor = processor.feature_extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        decoder = decoder
    elif decoder is None:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        decoder = None
    else:
        raise AssertionError
    return decoder, feature_extractor


@dataclass
class HfAsrPipelineFromLogitsInferencerDecoder(CachedData):
    """
    this one is a sister of Aschinglupi
    """

    logits_inferencer: HFWav2Vec2LogitsInferencer = UNDEFINED
    decoder: BaseCTCDecoder = UNDEFINED
    chunk_length_s: Optional[float] = None
    stride_length_s: Optional[Union[tuple[float, float], list[float]]] = None
    cache_base: PrefixSuffix = field(
        default_factory=lambda: BASE_PATHES["asr_inference"]
    )

    @property
    def name(self):
        # TODO: decoder has no name?
        return f"hfpipeline-{self.logits_inferencer.name}"

    def _build_cache(self):
        pass

    def _prepare_asr_pipeline(self):
        if isinstance(self.decoder, PyCTCKenLMDecoder):
            self.decoder.build()
            pyctc_decoder_or_None = self.decoder._pyctc_decoder
            assert pyctc_decoder_or_None is not None
        else:
            pyctc_decoder_or_None = None
        model_id = self.logits_inferencer.checkpoint.model_path
        decoder, feature_extractor = prepare_decoder_and_feature_extractor(
            pyctc_decoder_or_None, model_id
        )

        # sampling_rate = feature_extractor.sampling_rate
        # assert sampling_rate == self.sample_rate
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)

        self.device = 0 if torch.cuda.is_available() else -1
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=self.device,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            decoder=decoder,
        )
        if decoder is not None:
            assert asr_pipeline.type == "ctc_with_lm"
        return asr_pipeline

    def _post_build_setup(self):
        # self.logits_inferencer.build() # might be "consequential" but is not necessary here
        self.asr_pipeline = self._prepare_asr_pipeline()

    @beartype
    def predict(self, audio_array) -> str:
        audio_array = audio_array.astype(np.float)

        prediction = self.asr_pipeline(
            audio_array,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
        )
        hyp = prediction["text"]
        return hyp

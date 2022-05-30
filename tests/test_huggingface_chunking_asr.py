from dataclasses import dataclass, field
from itertools import islice
from math import floor, ceil
from time import time
from typing import Iterable, Optional

import numpy as np
import pytest
import torch
from beartype import beartype
from numpy.testing import assert_allclose

from conftest import get_test_cache_base, TEST_RESOURCES, get_test_vocab
from misc_utils.beartypes import TorchTensor2D, NumpyFloat2DArray
from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)
from ml4audio.audio_utils.overlap_array_chunker import AudioMessageChunk
from ml4audio.text_processing.ctc_decoding import (
    AlignedBeams,
    LogitAlignedTranscript,
)
from ml4audio.text_processing.lm_model_for_pyctcdecode import (
    KenLMForPyCTCDecodeFromArpa,
)
from ml4audio.text_processing.metrics_calculation import calc_cer
from ml4audio.audio_utils.audio_io import (
    load_resample_with_nemo,
)
from ml4audio.text_processing.asr_text_normalization import (
    normalize_filter_text,
    Casing,
    TranscriptNormalizer,
)
from ml4audio.text_processing.pretty_diff import smithwaterman_aligned_icdiff

from ml4audio.text_processing.pyctc_decoder import PyCTCKenLMDecoder, OutputBeamDc
from conftest import overlapping_audio_messages_from_audio_array

cache_base = get_test_cache_base()
BASE_PATHES["asr_inference"] = cache_base

StartEnd = tuple[int, int]


"""
@beartype
def hf_original_chunking_method(audio: NumpyFloat1DArray, chunk_len: int, step: int):
    
    based on: https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/src/transformers/pipelines/automatic_speech_recognition.py#L53
    
    inputs_len = audio.shape[0]
    stride_left = chunk_len - step
    for fi in range(0, inputs_len, step):
        chunk = audio[fi : fi + chunk_len]
        _stride_left = 0 if fi == 0 else stride_left
        is_last = fi + step >= inputs_len
        if chunk.shape[0] > _stride_left:
            yield {
                "is_last": is_last,
                "audio_slice": (fi, fi + len(chunk)),
                "chunk": chunk,
            }
"""


@dataclass
class StreamingDecoder(PyCTCKenLMDecoder):

    _buffer: NumpyFloat2DArray = field(init=False)
    _last_end: int = field(init=False, default=0)
    _lm_states: Optional[list["kenlm.State"]] = field(init=False, default=None)

    def reset(self):
        self._buffer = None
        self._last_end = 0
        self._lm_states = None

    @beartype
    def calc_left_right(
        self, logits: NumpyFloat2DArray, start_end: tuple[int, int]
    ) -> tuple[Optional[NumpyFloat2DArray], NumpyFloat2DArray]:
        """
        based on: https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/src/transformers/pipelines/automatic_speech_recognition.py#L337
        """
        a_start, a_end = start_end
        if self._last_end > 0:
            audio_slice_len = a_end - a_start
            logit_window_len = logits.shape[0]
            audio_to_logits_ratio = audio_slice_len / logit_window_len
            logits_overlap = (self._last_end - a_start) / audio_to_logits_ratio
            right_part = logits[floor(logits_overlap / 2) :]
            left_part = self._buffer[: -ceil(logits_overlap / 2)]

        else:
            right_part = logits
            left_part = None

        self._buffer = right_part
        self._last_end = a_end

        return left_part, right_part

    @beartype
    def decode(self, ctc_matrix: NumpyFloat2DArray) -> AlignedBeams:
        beams = [
            OutputBeamDc(*b)
            for b in self._pyctc_decoder.decode_beams(
                ctc_matrix,
                beam_width=self.beam_size,
                lm_start_state=self._lm_states,
            )
        ]
        self._lm_states = [b.last_lm_state for b in beams]

        return [
            LogitAlignedTranscript.create_from_token_spans(
                b.text_frames, b.logit_score, b.lm_score
            )
            for b in islice(beams, self.num_best)
        ]


@beartype
def _hf_postprocess(
    model_outputs: Iterable[tuple[TorchTensor2D, tuple[int, int]]], decoder
):
    """
    based on: https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/src/transformers/pipelines/automatic_speech_recognition.py#L337
    """
    ttype = "ctc"

    buffer = []
    last_end = 0
    for logits, (a_start, a_end) in model_outputs:
        # logits: TorchTensor2D
        audio_slice_len = a_end - a_start
        logit_window_len = logits.shape[0]
        audio_to_logits_ratio = audio_slice_len / logit_window_len
        if last_end > 0:
            logits_overlap = (last_end - a_start) / audio_to_logits_ratio
        else:
            logits_overlap = 0.0

        if len(buffer) > 0:
            buffer[-1] = buffer[-1][: -ceil(logits_overlap / 2)]

        buffer.append(logits[floor(logits_overlap / 2) :].numpy())
        last_end = a_end
    return buffer, decode_with_lm(decoder, buffer)


@beartype
def _greedy_decode(tokenizer, buffer: list[NumpyFloat2DArray]):
    items = np.concatenate(buffer, axis=0)
    items = items.argmax(axis=-1)

    skip_special_tokens = False
    text = tokenizer.decode(items, skip_special_tokens=skip_special_tokens)
    return text


def decode_with_lm(decoder, buffer):
    logits = np.concatenate(buffer, axis=0)
    transcript = decoder.decode(torch.from_numpy(logits.squeeze()))[0]

    hyp = transcript.text
    return hyp


def _hf_forward(inferencer: HFWav2Vec2LogitsInferencer, ac: AudioMessageChunk):
    logits = inferencer.resample_calc_logits(ac.audio_array)
    start_end = (ac.frame_idx, ac.frame_idx + len(ac.audio_array))
    return logits, start_end


@pytest.mark.skip("FIXME")
@pytest.mark.parametrize(
    "step_dur,window_dur,max_CER",
    [
        # (1.0, 2.0, 0.052),
        (1.0, 4.0, 0.0099),
        # (4.0, 8.0, 0.0065),
        # (1.0, 8.0, 0.023),
    ],
)
def test_HF_chunking_asr(
    hfwav2vec2_base_logits_inferencer: HFWav2Vec2LogitsInferencer,
    librispeech_audio_file,
    librispeech_raw_ref,
    vocab,
    step_dur: float,
    window_dur: float,
    max_CER: float,
):

    expected_sample_rate = hfwav2vec2_base_logits_inferencer.input_sample_rate
    audio_array = load_resample_with_nemo(librispeech_audio_file, expected_sample_rate)

    start_time = time()
    tokenizer = hfwav2vec2_base_logits_inferencer._processor.tokenizer
    from packaging import version

    inference_context = (
        torch.inference_mode
        if version.parse(torch.__version__) >= version.parse("1.9.0")
        else torch.no_grad
    )

    audio_chunks_g = overlapping_audio_messages_from_audio_array(
        audio_array=audio_array,
        sr=expected_sample_rate,
        chunk_dur=window_dur,
        step_dur=step_dur,
    )
    startup_time = time() - start_time
    start_time = time()
    with inference_context():
        forward_oupt = [
            _hf_forward(
                hfwav2vec2_base_logits_inferencer,
                ac,
            )
            for ac in audio_chunks_g
        ]
    decoder, stream_decoder = _prepare_decoder(tokenizer)
    left_right = [
        stream_decoder.calc_left_right(l.numpy(), s_e) for l, s_e in forward_oupt
    ]
    print(f"{[(l.shape if l is not None else 0,r.shape) for l,r in left_right ]=}")
    parts = [l for l, r in left_right] + [left_right[-1][1]]
    parts = [x for x in parts if x is not None]

    buffer, hyp = _hf_postprocess(forward_oupt, decoder)
    logits = np.concatenate(buffer, axis=0)

    assert_allclose(np.concatenate(parts), logits)

    hyp = "".join(
        [stream_decoder.decode(logits_chunk)[0].text for logits_chunk in parts]
    )
    inference_duration = time() - start_time

    ref = normalize_filter_text(
        librispeech_raw_ref,
        vocab,
        text_normalizer="en",
        casing=Casing.upper,
    )
    diff_line = smithwaterman_aligned_icdiff(ref, hyp)
    print(f"{window_dur=},{step_dur=}")

    print(diff_line)
    cer = calc_cer([(hyp, ref)])
    print(
        f"CER: {cer},start-up took: {startup_time}, inference took: {inference_duration} seconds"
    )
    assert max_CER > cer


def _prepare_decoder(tokenizer):
    tn = TranscriptNormalizer(
        casing=Casing.upper, text_normalizer="en", vocab=get_test_vocab()
    )
    arpa = KenLMForPyCTCDecodeFromArpa(
        name="test",
        cache_base=cache_base,
        arpa_file=f"{TEST_RESOURCES}/lm.arpa",
        transcript_normalizer=tn,
    )
    decoder = PyCTCKenLMDecoder(
        tokenizer=tokenizer,
        lm_weight=1.0,
        beta=0.5,
        lm_data=arpa,
    )
    decoder.build()
    stream_decoder = StreamingDecoder(
        tokenizer=tokenizer,
        lm_weight=1.0,
        beta=0.5,
        lm_data=arpa,
    )
    stream_decoder.build()
    return decoder, stream_decoder

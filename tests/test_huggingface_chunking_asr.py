from dataclasses import dataclass
from math import floor, ceil
from time import time
from typing import Iterable

import numpy as np
import pytest
import torch
from beartype import beartype

from conftest import get_test_cache_base, TEST_RESOURCES, get_test_vocab
from misc_utils.beartypes import NumpyFloat1DArray, TorchTensor2D, NumpyFloat2DArray
from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)
from ml4audio.text_processing.ctc_decoding import BaseCTCDecoder
from ml4audio.text_processing.lm_model_for_pyctcdecode import (
    KenLMForPyCTCDecodeFromArpa,
)
from ml4audio.text_processing.metrics_calculation import calc_cer
from ml4audio.audio_utils.audio_io import (
    load_resample_with_nemo,
    break_array_into_chunks,
    convert_to_16bit_array,
)
from ml4audio.audio_utils.audio_message_chunking import (
    AudioMessageChunker,
    audio_messages_from_chunks,
    AudioMessageChunk,
)
from ml4audio.text_processing.asr_text_normalization import (
    normalize_filter_text,
    Casing,
    TranscriptNormalizer,
)
from ml4audio.text_processing.pretty_diff import smithwaterman_aligned_icdiff

from ml4audio.text_processing.pyctc_decoder import PyCTCKenLMDecoder

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


@beartype
def hf_chunking_preprocessing(
    audio_array: NumpyFloat1DArray, sr: int, step_dur: float, chunk_dur: float
):

    chunker = AudioMessageChunker(
        chunk_size=int(chunk_dur * sr),
        min_step_size=int(step_dur * sr),
    )
    chunker.reset()

    audio_array = convert_to_16bit_array(audio_array)
    small_chunks = break_array_into_chunks(audio_array, int(sr * 0.1))
    chunks_g = (
        am
        for ch in audio_messages_from_chunks("dummy-id", small_chunks)
        for am in chunker.handle_datum(ch)
    )
    yield from chunks_g


# @dataclass
# class StreamingDecoder:
#
#     decoder: PyCTCKenLMDecoder
#     _buffer: NumpyFloat2DArray
#     _start_end: tuple[int, int] = (0, 0)
#
#     def reset(self):
#         self._buffer = np.zeros((0, 0), dtype=np.float)
#
#     @beartype
#     def decode(self, logits: NumpyFloat2DArray, start_end: tuple[int, int]):
#         """
#         based on: https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/src/transformers/pipelines/automatic_speech_recognition.py#L337
#         """
#         a_start, a_end = start_end
#         last_end = 0
#         # logits: TorchTensor2D
#         audio_slice_len = a_end - a_start
#         logit_window_len = logits.shape[0]
#         audio_to_logits_ratio = audio_slice_len / logit_window_len
#         if last_end > 0:
#             logits_overlap = (last_end - a_start) / audio_to_logits_ratio
#         else:
#             logits_overlap = 0.0
#
#         right_part = logits[floor(logits_overlap / 2) :]
#
#         if len(self._buffer) > 0:
#             left_part = self._buffer[: -ceil(logits_overlap / 2)]
#             self._buffer = np.concatenate(
#                 [left_part, right_part]
#             )
#         else:
#             self._buffer = right_part
#
#         last_end = a_end
#         return _greedy_decode(tokenizer, self._buffer)


@beartype
def _hf_postprocess(
    model_outputs: Iterable[tuple[TorchTensor2D, tuple[int, int]]], tokenizer
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
    return decode_with_lm(tokenizer, buffer)


@beartype
def _greedy_decode(tokenizer, buffer: list[NumpyFloat2DArray]):
    items = np.concatenate(buffer, axis=0)
    items = items.argmax(axis=-1)

    skip_special_tokens = False
    text = tokenizer.decode(items, skip_special_tokens=skip_special_tokens)
    return text


def decode_with_lm(tokenizer, buffer):
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

    logits = np.concatenate(buffer, axis=0)
    transcript = decoder.decode(torch.from_numpy(logits.squeeze()))[0]

    hyp = transcript.text
    return hyp


def _hf_forward(inferencer: HFWav2Vec2LogitsInferencer, ac: AudioMessageChunk):
    logits = inferencer.resample_calc_logits(ac.audio_array)
    start_end = (ac.frame_idx, ac.frame_idx + len(ac.audio_array))
    return logits, start_end


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

    audio_chunks_g = hf_chunking_preprocessing(
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
    hyp = _hf_postprocess(forward_oupt, tokenizer)
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

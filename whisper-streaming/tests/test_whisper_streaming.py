from typing import Optional

import numpy as np
import pytest

from ml4audio.asr_inference.faster_whisper_inferencer import (
    FasterWhisperASRSegmentInferencer,
    FasterWhisperArgs,
)
from ml4audio.asr_inference.inference import StartEndTextsNonOverlap
from ml4audio.audio_utils.overlap_array_chunker import (
    audio_messages_from_file,
    OverlapArrayChunker,
)
from ml4audio.text_processing.asr_metrics import calc_cer
from ml4audio.text_processing.asr_text_cleaning import (
    VocabCasingAwareTextCleaner,
    Casing,
)
from ml4audio.text_processing.pretty_diff import smithwaterman_aligned_icdiff
from whisper_streaming.whisper_streaming import WhisperStreamer


@pytest.mark.parametrize(
    "step_dur,window_dur,max_CER,num_responses_expected",
    [
        # fmt: off
        (4.0, 4.0, 0.053, 7),
        # fmt: on
    ],
)
def test_whisper_streaming(
    librispeech_audio_file,
    librispeech_ref,
    step_dur: float,
    window_dur: float,
    max_CER: float,
    num_responses_expected: int,
):
    inferencer = FasterWhisperASRSegmentInferencer(
        model_name="base", whisper_args=FasterWhisperArgs(language="en")
    )
    inferencer.build()
    SR = inferencer.sample_rate
    asr_input = list(
        audio_messages_from_file(librispeech_audio_file, SR, chunk_duration=0.1)
    )
    assert asr_input[-1].end_of_signal
    audio_signal = np.concatenate([ac.array for ac in asr_input])
    wav_length = 393920
    opus_is_alittle_longer = 70
    assert audio_signal.shape[0] == wav_length + opus_is_alittle_longer

    streaming_asr: WhisperStreamer = WhisperStreamer(
        asr_inferencer=inferencer,
        audio_bufferer=OverlapArrayChunker(
            chunk_size=int(window_dur * SR),
            # minimum_chunk_size=int(1 * SR),  # one second!
            min_step_size=int(step_dur * SR),
            # max_step_size=int(max_step_dur * SR) if max_step_dur is not None else None,
        ),
    )
    streaming_asr.build()
    segments: StartEndTextsNonOverlap = []
    num_responses = 0
    with streaming_asr:
        for inpt in asr_input:
            for new_segments in streaming_asr.handle_inference_input(inpt):
                num_responses += 1
                # print(f"{new_segments=}")
                segments = [
                    (s, e, t) for s, e, t in segments if e <= new_segments[0][0]
                ] + new_segments
                print(f"{segments=}")
    assert num_responses_expected == num_responses, f"{num_responses=}"
    hyp = " ".join(t for _, _, t in segments)
    cleaner = VocabCasingAwareTextCleaner(
        casing=Casing.upper,
        text_cleaner_name="en",
        letter_vocab=list(set(librispeech_ref)),
    )
    hyp = cleaner(hyp)
    ref = librispeech_ref
    print(smithwaterman_aligned_icdiff(ref, hyp))
    cer = calc_cer([ref], [hyp])
    print(f"{step_dur=},{window_dur=},{cer=}")

    assert cer <= max_CER

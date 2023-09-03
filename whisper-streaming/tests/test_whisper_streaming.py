import numpy as np
import pytest
from beartype import beartype

from ml4audio.asr_inference.faster_whisper_inferencer import (
    FasterWhisperArray2SegmentedTranscripts,
    FasterWhisperArgs,
)
from ml4audio.asr_inference.inference import StartEndTextsNonOverlap
from ml4audio.audio_utils.overlap_array_chunker import (
    OverlapArrayChunker,
)
from ml4audio.audio_utils.audio_io import audio_messages_from_file
from ml4audio.text_processing.asr_metrics import calc_cer
from ml4audio.text_processing.asr_text_cleaning import (
    VocabCasingAwareTextCleaner,
    Casing,
)
from ml4audio.text_processing.pretty_diff import smithwaterman_aligned_icdiff
from whisper_streaming.whisper_streaming import (
    WhisperStreamer,
    concat_transcript,
    OverlappingSegment,
    accumulate_transcript,
)


@pytest.mark.parametrize(
    "name,step_dur,window_dur,max_CER,num_responses_expected",
    [
        # fmt: off
        # one might be tempted to interpret those CER-values, but I think there is not pattern here, everything around 5% or lower is acceptable, 3% is NOT better than 5%! its just a very-small+very instable "base-whisper"-model!
        ("non-overlapping",4.0, 4.0, 0.045, 7),
        ("good-overlap",2.0, 4.0, 0.042, 12),
        ("big-overlap",1.0, 4.0, 0.05, 22),
        # fmt: on
    ],
)
def test_whisper_streaming(
    name: str,
    librispeech_audio_file: str,
    librispeech_ref: str,
    step_dur: float,
    window_dur: float,
    max_CER: float,
    num_responses_expected: int,
):
    inferencer = FasterWhisperArray2SegmentedTranscripts(
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
    audio_len = audio_signal.shape[0]
    print(f"audio-dur: {audio_len/16000}")
    assert audio_len == wav_length + opus_is_alittle_longer

    streaming_asr: WhisperStreamer = WhisperStreamer(
        asr_inferencer=inferencer,
        audio_bufferer=OverlapArrayChunker(
            chunk_size=int(window_dur * SR),
            # minimum_chunk_size=int(1 * SR),  # one second!
            min_step_size=int(step_dur * SR),
            # max_step_size=int(max_step_dur * SR) if max_step_dur is not None else None,
        ),
        overwrite_last_k_words=3,
    )
    streaming_asr.build()
    transcript: str = ""
    num_responses = 0
    with streaming_asr:
        for inpt in asr_input:
            for overlap_segment, new_segments in streaming_asr.handle_inference_input(
                inpt
            ):
                num_responses += 1
                transcript = accumulate_transcript(
                    overlap_segment, new_segments, transcript
                )
                print(f"{overlap_segment=}###{new_segments=}")
    assert "  " not in transcript
    hyp = transcript

    cleaner = VocabCasingAwareTextCleaner(
        casing=Casing.upper,
        text_cleaner_name="en",
        letter_vocab=list(set(librispeech_ref)),
    )
    hyp = cleaner(hyp)
    ref = librispeech_ref
    print(smithwaterman_aligned_icdiff(ref, hyp))
    cer = calc_cer([ref], [hyp])
    print(f"{name}: {step_dur=},{window_dur=},{cer=}")
    assert cer <= max_CER
    assert num_responses_expected == num_responses, f"{num_responses=}"

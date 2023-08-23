from dataclasses import dataclass
from typing import Annotated, Optional, Union

from beartype import beartype
from beartype.vale import Is

from misc_utils.beartypes import NumpyFloat1D, NeList
from misc_utils.buildable_data import BuildableData
from ml4audio.asr_inference.inference import (
    ASRAudioSegmentInferencer,
    StartEndTextsNonOverlap,
)
from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEnd,
    fix_segments_to_non_overlapping,
)

MINIMAL_SEGMENT_LENGHTS = 0.08  # TODO: this is super arbitray!


@beartype
def fix_start_end(start_end: tuple[float, float], audio_dur: float) -> StartEnd:
    """
    TODO
        13.06. I was changing audio_dur to "Optional[float]=None" -> why?
        somehow I had an issue fix unfixable whisper-segments, multiple in a row being empty or something
    """
    start, end = start_end
    if start < 0:
        print(f"WTF! whisper gave {start=}")
        start = 0.0

    # if end > audio_dur: # TODO: trying to correct "after-audio-hallucinations" like this is not working!
    #     print(f"WTF! whisper gave {end=} that is after {audio_dur=} -> {audio_dur=}")
    #     # if end-audio_dur>10.0:
    #     #     raise AssertionError(f"thats too much! cannot fix it!")
    #     end = audio_dur

    if end - start < MINIMAL_SEGMENT_LENGHTS:
        print(f"WTF! whisper gave {(start,end)=}")
        start = end - MINIMAL_SEGMENT_LENGHTS
        # end = min(audio_dur, start + 0.04)

    return (start, end)


@beartype
def fix_whisper_segments(
    whisper_segments: NeList[tuple[float, float, str]], audio_dur: float
) -> StartEndTextsNonOverlap:

    start_end = [
        fix_start_end((s, e), audio_dur)
        for s, e, text in whisper_segments
        if len(text) > 0
        # if s < audio_dur # TODO: one could filter for potentially hallucinated like this
    ]
    # start_ends_merged=[]
    # for s,e,t in zip(start_end,whisper_segments):

    start_end = fix_segments_to_non_overlapping(start_end)
    return [
        (start, end, text)
        for (_, _, text), (start, end) in zip(whisper_segments, start_end)
    ]


@dataclass(frozen=True)
class WhisperArgs:
    task: Annotated[str, Is[lambda s: s in WHISPER_TASKS]]
    language: str = "de"
    temperature: Optional[Union[float, tuple[float, ...], list[float]]] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )  # this is default in whisper code
    # don't mess with the temperatures! they are needed for fallback if beam-search fails!
    beam_size: Optional[int] = None  # default=5 see whisper code


WHISPER_TASKS = {"transcribe", "translate"}


@dataclass
class WhisperInferencer(BuildableData, ASRAudioSegmentInferencer):
    whisper_args: Optional[WhisperArgs] = None

    @beartype
    def predict_transcribed_segments(
        self, audio_array: NumpyFloat1D
    ) -> StartEndTextsNonOverlap:
        return self.predict_transcribed_with_whisper_args(
            audio_array, self.whisper_args
        )

    @beartype
    def predict_transcribed_with_whisper_args(
        self, audio_array: NumpyFloat1D, whisper_args: WhisperArgs
    ) -> StartEndTextsNonOverlap:
        raise NotImplementedError

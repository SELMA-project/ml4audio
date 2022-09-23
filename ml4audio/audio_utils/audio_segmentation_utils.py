import os
from collections import namedtuple
from typing import Annotated, Iterable

import soundfile
from beartype import beartype
from beartype.door import die_if_unbearable
from beartype.vale import Is
from tqdm import tqdm

from misc_utils.beartypes import NumpyFloat1DArray, NeList, is_bearable
from ml4audio.asr_inference.transcript_glueing import NonEmptyAlignedTranscript


non_empty = lambda x: x[1] > x[0]
start_non_negative = lambda x: x[0] >= 0
StartEndIdx = Annotated[
    tuple[int, int],
    (Is[non_empty] & Is[start_non_negative]),
]


# naming lamdas cause it somehow helps beartype, see: https://github.com/beartype/beartype/blob/305d73792de59d8f9918fabaab76203402ddb8c6/beartype/_util/func/utilfunccode.py#L348
StartEnd = Annotated[
    tuple[float, float],
    (Is[non_empty] & Is[start_non_negative]),
]
TimeSpan = StartEnd  # rename?

valid_label = lambda x: len(x[2]) > 0
StartEndLabel = Annotated[
    tuple[float, float, str],
    (Is[non_empty] & Is[start_non_negative] & Is[valid_label]),
]
StartEndLabels = NeList[StartEndLabel]


def is_non_overlapping(seq: NeList[StartEnd]) -> bool:
    return all((seq[k - 1][1] <= seq[k][0] for k in range(1, len(seq))))


NonOverlSegs = Annotated[NeList[StartEnd], Is[is_non_overlapping]]


def is_weakly_monoton_increasing(seq: NeList[StartEnd]) -> bool:
    return all(seq[k - 1][0] <= seq[k][0] for k in range(1, len(seq)))


@beartype
def expand_merge_segments(
    segments: Annotated[NeList[StartEnd], Is[is_weakly_monoton_increasing]],
    max_gap_dur: float = 0.2,  # gap within a segment -> shorter than this gets merged
    expand_by: Annotated[float, Is[lambda x: x > 0]] = 0.1,
) -> NeList[StartEnd]:
    exp_segs: list[tuple[float, float]] = []
    prev_start: int = -9999
    prev_end: int = -9999
    for start, end in tqdm(segments, f"expand_merge_segments"):
        start -= expand_by
        end += expand_by

        is_expandable = len(exp_segs) > 0 and start - prev_end < max_gap_dur
        if is_expandable:
            startend = prev_start, end
            if not is_bearable(startend, StartEnd):
                print(f"{startend=}")
            die_if_unbearable(startend, StartEnd)
            exp_segs[-1] = startend
        else:
            startend = (start, end)
            die_if_unbearable(startend, StartEnd)
            exp_segs.append(startend)
        prev_start, prev_end = exp_segs[-1]

    assert all((e > s for s, e in exp_segs))
    return exp_segs


@beartype
def merge_short_segments(
    segments: NeList[tuple[float, float]], min_dur: float = 1.5
) -> NeList[StartEnd]:
    GIVE_ME_NEW_START = "<GIVE_ME_NEW_START>"

    def buffer_segment(segs: Iterable[tuple[float, float]]):
        buffer_start = GIVE_ME_NEW_START
        for start, end in segs:
            if buffer_start == GIVE_ME_NEW_START:
                buffer_start = start

            if end - buffer_start > min_dur:
                yield buffer_start, end
                buffer_start = GIVE_ME_NEW_START

    min_dur_segs = list(buffer_segment(segments))
    assert all((e - s > min_dur for s, e in min_dur_segs))
    return min_dur_segs


@beartype
def pause_based_segmentation(
    timestamped_letters: NeList[tuple[str, float]],
    min_seg_dur=1.5,
    max_gap_dur=0.2,
    expand_by=0.1,
) -> NeList[StartEnd]:
    _, timestamps = [list(x) for x in zip(*timestamped_letters)]
    letter_duration = (
        0.04  # heuristic -> 40ms is median of some transcript, sounds plausible!
    )
    timestamps = sorted(timestamps)  # god dammit!
    weakly_monoton_increasing = all(
        (timestamps[k + 1] - timestamps[k] >= 0.0 for k in range(len(timestamps) - 1))
    )
    assert weakly_monoton_increasing
    s_e_times = [(ts, ts + letter_duration) for k, ts in enumerate(timestamps)]
    s_e_times = expand_merge_segments(
        s_e_times, max_gap_dur=max_gap_dur, expand_by=expand_by
    )
    s_e_times = merge_short_segments(s_e_times, min_dur=min_seg_dur)
    return s_e_times


@beartype
def write_segmentwise_wav_file_just_for_fun(
    start_end_speaker: list[tuple[float, float, str]],
    array: NumpyFloat1DArray,
    SR=16000,
):
    output_dir = "segments_wavs"
    os.makedirs(output_dir, exist_ok=True)
    for k, (s, e, sp) in enumerate(start_end_speaker):
        soundfile.write(
            f"{output_dir}/{k}_{sp}.wav",
            array[round(s * SR) : round(e * SR)],
            samplerate=SR,
        )


# # pyannote Segment allows end<start -> wtf!
# seg=Segment(start=1,end=0.5)
# print(f"{seg.duration=},{seg=}")

if __name__ == "__main__":
    print(is_bearable((0.0, 1.0, "bo"), StartEndLabel))

import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Annotated, Iterable, Union

import soundfile
from beartype import beartype
from beartype.door import die_if_unbearable, is_bearable
from beartype.vale import Is

from misc_utils.beartypes import NumpyFloat1DArray, NeList

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


# purposefully not type-hint here! why not?
def is_non_overlapping(seq: list[tuple[float, float]]) -> bool:
    if len(seq) > 1:
        is_fine = all((seq[k - 1][1] <= seq[k][0] for k in range(1, len(seq))))
    else:
        is_fine = True
    return is_fine


StartEndLabelNonOverlap = Annotated[
    StartEndLabels,
    Is[is_non_overlapping],
]

NonOverlSegs = Annotated[NeList[StartEnd], Is[is_non_overlapping]]


def is_weakly_monoton_increasing(seq: NeList[StartEnd]) -> bool:
    if len(seq) > 1:
        is_fine = all(seq[k - 1][0] <= seq[k][0] for k in range(1, len(seq)))
    else:
        is_fine = True
    return is_fine


@beartype
def groups_to_merge_segments_of_same_label(
    start_end_label: NeList[StartEndLabel], max_gap_dur: float
) -> NeList[NeList[int]]:
    initial_group = [0]
    mergable_groupds = [initial_group]
    for k in range(len(start_end_label) - 1):
        start, end, label = start_end_label[k]
        next_start, next_end, next_label = start_end_label[k + 1]
        if label == next_label and next_start - end < max_gap_dur:
            mergable_groupds[-1].append(k + 1)
        else:
            new_group = [k + 1]
            mergable_groupds.append(new_group)

    return mergable_groupds


@beartype
def merge_segments_labelaware(
    start_end_label: NeList[StartEndLabel], max_gap_dur: float
) -> StartEndLabelNonOverlap:
    groups = groups_to_merge_segments_of_same_label(start_end_label, max_gap_dur)

    def merge_segment(first: int, last: int):
        group_start = start_end_label[first][0]
        group_end = start_end_label[last][1]
        group_label = start_end_label[first][2]
        return group_start, group_end, group_label

    merged = [merge_segment(gr[0], gr[-1]) for gr in groups]
    return merged


@dataclass
class Span:
    start: float
    end: float


def is_overlapping(span: Span, next_span: Span):
    return span.end > next_span.start


@beartype
def get_contiguous_stamps(
    start_ends: NeList[StartEnd],
) -> NonOverlSegs:
    """
    based on: get_contiguous_stamps from https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/nemo/collections/asr/parts/utils/speaker_utils.py#L230
    """
    cont_spans = [Span(s, e) for s, e in start_ends]

    overlapping_idx = filter(
        lambda i: is_overlapping(cont_spans[i], cont_spans[i + 1]),
        range(len(start_ends) - 1),
    )
    for i in overlapping_idx:
        avg = (cont_spans[i].start + cont_spans[i + 1].end) / 2.0
        cont_spans[i].end = avg
        cont_spans[i + 1].start = avg

    return [(x.start, x.end) for x in cont_spans]


@beartype
def expand_segments(
    segments: Annotated[NeList[StartEnd], Is[is_weakly_monoton_increasing]],
    expand_by: Annotated[float, Is[lambda x: x > 0]] = 0.1,
) -> NonOverlSegs:
    raw_expaned = [(start - expand_by, end + expand_by) for start, end in segments]
    return get_contiguous_stamps(raw_expaned)


@beartype
def expand_merge_segments_labelaware(
    start_end_labels: Annotated[
        NeList[StartEndLabel], Is[is_weakly_monoton_increasing]
    ],
    expand_by: float,
    max_gap_dur: float,
) -> StartEndLabelNonOverlap:
    """
    used for post-clustering resegmentation
    """
    s_e_exp = expand_segments(
        [(s, e) for s, e, _ in start_end_labels], expand_by=expand_by
    )
    s_e_labels = merge_segments_labelaware(
        [(s, e, l) for (s, e), (_, _, l) in zip(s_e_exp, start_end_labels)],
        max_gap_dur=max_gap_dur,
    )
    return s_e_labels


NonOverlappingMonotonIncreasingSegments = Annotated[
    NeList[StartEnd], Is[is_non_overlapping] & Is[is_weakly_monoton_increasing]
]


@beartype
def expand_merge_segments(
    segments: Annotated[NeList[StartEnd], Is[is_weakly_monoton_increasing]],
    max_gap_dur: float = 0.2,  # gap within a segment -> shorter than this gets merged
    expand_by: Annotated[float, Is[lambda x: x > 0]] = 0.1,
) -> NonOverlappingMonotonIncreasingSegments:
    exp_segs: list[tuple[float, float]] = []
    prev_start: int = -9999
    prev_end: int = -9999
    for start, end in expand_segments(segments, expand_by):

        is_expandable = len(exp_segs) > 0 and start - prev_end < max_gap_dur
        if is_expandable:
            startend = prev_start, end
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
    segments: NonOverlappingMonotonIncreasingSegments, min_dur: float = 1.5
) -> NonOverlappingMonotonIncreasingSegments:
    GIVE_ME_NEW_START = -1

    def buffer_segment(segs: Iterable[tuple[float, float]]):
        previous_start: float = GIVE_ME_NEW_START
        for start, end in segs:
            if previous_start == GIVE_ME_NEW_START:
                previous_start = start

            if end - previous_start > min_dur:
                yield previous_start, end
                previous_start = GIVE_ME_NEW_START

        vield_very_last = previous_start != GIVE_ME_NEW_START
        if vield_very_last:
            yield previous_start, end

    min_dur_segs: list = list(buffer_segment(segments))
    last_start, last_end = min_dur_segs[-1]
    if last_end - last_start < min_dur and len(min_dur_segs) > 1:
        _, last_end = min_dur_segs.pop(-1)
        min_dur_segs[-1] = (min_dur_segs[-1][0], last_end)

    return min_dur_segs


def is_weakly_monoton_increasing_timeseries(timestamps: list[float]) -> bool:
    return all(
        (timestamps[k + 1] - timestamps[k] >= 0.0 for k in range(len(timestamps) - 1))
    )


@beartype
def segment_letter_timestamps(
    letter_timestamps: Annotated[
        NeList[float], Is[is_weakly_monoton_increasing_timeseries]
    ],
    min_seg_dur=1.5,
    max_gap_dur=0.2,
    expand_by=0.1,
) -> NonOverlappingMonotonIncreasingSegments:
    letter_duration = (
        0.04  # heuristic -> 40ms is median of some transcript, sounds plausible!
    )
    s_e_times = [(ts, ts + letter_duration) for ts in letter_timestamps]
    s_e_times = expand_merge_segments(
        s_e_times, max_gap_dur=max_gap_dur, expand_by=expand_by
    )
    s_e_times = merge_short_segments(s_e_times, min_dur=min_seg_dur)

    def check_segment_validity():
        first_ts = letter_timestamps[0]
        assert s_e_times[0][0] == first_ts - expand_by, f"{s_e_times[0][0]=},{first_ts}"
        last_ts = letter_timestamps[-1]
        assert (
            s_e_times[-1][1] == last_ts + letter_duration + expand_by
        ), f"{s_e_times[-1][1]=},{last_ts}"

    check_segment_validity()
    return s_e_times


@beartype
def write_segmentwise_wav_file_just_for_fun(
    start_end_speaker: list[tuple[float, float, str]],
    array: NumpyFloat1DArray,
    SR: int = 16000,
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

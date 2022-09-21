import os
from collections import namedtuple
from typing import Annotated, Iterable

import soundfile
from beartype import beartype
from beartype.door import die_if_unbearable
from beartype.vale import Is

from misc_utils.beartypes import NumpyFloat1DArray, NeList, is_bearable
from ml4audio.asr_inference.transcript_glueing import NonEmptyAlignedTranscript


@beartype
def expand_segments(
    s_e: list[tuple[int, int]], timestamps: list[float]
) -> list[tuple[float, float]]:
    """
    TODO: not used anymore?
    """

    def expand_to_the_right(k: int, this_end: float, expand_by=0.1):
        if k + 1 <= len(s_e) - 1:
            next_start, _ = s_e[k + 1]
            if next_start < len(timestamps):
                dur_to_next_start = timestamps[next_start] - this_end
                expand_by = min(expand_by, dur_to_next_start / 2)
        return expand_by

    def expand_to_the_left(k: int, this_start: float, expand_by=0.1):
        if k > 0:
            _, last_end = s_e[k - 1]
            dur_to_last_end = this_start - timestamps[last_end]
            expand_by = min(expand_by, dur_to_last_end / 2)
        return expand_by

    expand_by = 0.1
    s_e_times = [
        (
            timestamps[s] - expand_to_the_left(k, timestamps[s], expand_by=expand_by),
            timestamps[e] + expand_to_the_right(k, timestamps[e], expand_by=expand_by),
        )
        for k, (s, e) in enumerate(s_e)
    ]
    # print(f"{at.offset}")
    # print([f"{l.letter} -> {at.abs_timestamp(l):.3f}" for l in at.letters])

    return s_e_times


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


@beartype
def expand_merge_segments(
    segments: NeList[tuple[float, float]],
    min_gap_dur: float = 0.2,  # shorter than this gets merged
    expand_by: Annotated[float, Is[lambda x: x > 0]] = 0.1,
) -> NeList[StartEnd]:
    exp_segs: list[tuple[float, float]] = []
    for start, end in segments:
        start -= expand_by
        end += expand_by
        if end <= start:
            end = start + 0.1  # TODO: WTF!

        if len(exp_segs) > 0:
            prev_start, prev_end = exp_segs[-1]
        else:
            prev_start, prev_end = None, -9999

        if start - prev_end < min_gap_dur:
            startend = prev_start, end
            die_if_unbearable(startend, StartEnd)
            exp_segs[-1] = startend
        else:
            startend = (start, end)
            die_if_unbearable(startend, StartEnd)
            exp_segs.append(startend)

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
def pause_segmented_idx(
    timestamped_letters: NeList[tuple[str, float]], min_pause_dur: float = 0.5
) -> NeList[tuple[int, int]]:
    """
    indizes can be start == end
    """
    letters, timestamps = [list(x) for x in zip(*timestamped_letters)]
    text = "".join(letters)

    def calc_pause(k):
        if text[k - 1] == " ":
            previous = k - 2
        else:
            previous = k - 1
        pause_dur = timestamps[k] - timestamps[previous]
        return previous, k, pause_dur

    start_end_pausedur = [calc_pause(k) for k in range(1, len(text)) if text[k] != " "]
    Seg = namedtuple("startend", ["start", "end"])
    pause_segments = [
        Seg(start, end)
        for start, end, pause_dur in start_end_pausedur
        if pause_dur > min_pause_dur
    ]
    segments = (
        [(0, pause_segments[0].start)]
        + [
            (
                pause_segments[k - 1].end,
                pause_segments[k].start,
            )
            for k in range(1, len(pause_segments))
        ]
        + [(pause_segments[-1].end, len(text) - 1)]
    )
    return segments


@beartype
def pause_based_segmentation(
    timestamped_letters: NeList[tuple[str, float]],
    min_pause_dur=0.7,
    min_seg_dur=1.5,
    min_gap_dur=0.2,
    expand_by=0.1,
) -> NeList[StartEnd]:
    s_e = pause_segmented_idx(
        timestamped_letters,
        min_pause_dur=min_pause_dur,
    )
    _, timestamps = [list(x) for x in zip(*timestamped_letters)]
    monoton_increasing = all(
        (
            abs(timestamps[k] - timestamps[k - 1]) >= 0.0
            for k in range(len(timestamps) - 1)
        )
    )
    assert monoton_increasing
    s_e_times = [(timestamps[s], timestamps[e]) for s, e in s_e]
    s_e_times = expand_merge_segments(
        s_e_times, min_gap_dur=min_gap_dur, expand_by=expand_by
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

import os
from collections import namedtuple

import soundfile
from beartype import beartype

from misc_utils.beartypes import NumpyFloat1DArray
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript


@beartype
def expand_segments(
    s_e: list[tuple[int, int]], timestamps: list[float]
) -> list[tuple[float, float]]:
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


def expand_merge_segments(
    segments: list[tuple[float, float]],
    max_gap_dur: float = 0.2,
    expand_by: float = 0.1,
) -> list[tuple[float, float]]:
    overlap_stamps: list[tuple[float, float]] = []
    for start, end in segments:
        start -= expand_by
        end += expand_by
        if len(overlap_stamps) > 0:
            prev_start, prev_end = overlap_stamps[-1]
        else:
            prev_start, prev_end = None, -9999

        if start - prev_end < max_gap_dur:
            overlap_stamps[-1] = prev_start, prev_end
        else:
            overlap_stamps.append((start, end))

    return overlap_stamps


@beartype
def segment_by_pauses(
    at: AlignedTranscript, min_pause_dur: float = 0.5
) -> list[tuple[int, int]]:
    timestamps = at.rel_timestamps
    text = at.text

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
    # pauses_dur = [
    #     (
    #         f"{at.abs_timestamps[s]:.1f}",
    #         at.text[(s - 5) : (e + 5)],
    #         at.text[s : (e + 1)],
    #     )
    #     for s, e in pause_segments
    # ]
    segments = (
        [(0, pause_segments[0].start)]
        + [
            (
                pause_segments[k - 1].end,
                pause_segments[k].start,
            )
            for k in range(1, len(pause_segments))
        ]
        + [(pause_segments[-1].end, len(at.letters) - 1)]
    )
    return segments


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

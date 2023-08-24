import os
import sys
from dataclasses import dataclass

from beartype import beartype

from misc_utils.utils import Singleton
from ml4audio.audio_utils.aligned_transcript import (
    TimestampedLetters,
)

sys.path.append(".")
import difflib
from typing import Union, Iterable

import numpy as np


DEBUG = os.environ.get("DEBUG_GLUER", "False").lower() != "false"
if DEBUG:
    print("DEBUGGING MODE for glue_left_right")



"""
───▄▄▄
─▄▀░▄░▀▄
─█░█▄▀░█
─█░▀▄▄▀█▄█▄▀
▄▄█▄▄▄▄███▀

"""


@dataclass
class _NO_NEW_SUFFIX(metaclass=Singleton):
    """
    I guess this is a dataclass to enable serialization?
    """

    pass


NO_NEW_SUFFIX = _NO_NEW_SUFFIX()


@beartype
def calc_new_suffix(
    left: TimestampedLetters,
    right: TimestampedLetters,
    sm: difflib.SequenceMatcher,
) -> Union[TimestampedLetters, _NO_NEW_SUFFIX]:
    """
    two overlapping sequences

    left:_____----:-
    right:_______-:----

    colon is glue-point which is to be found

    return new suffix
    """
    is_overlapping = left.timestamps[-1] > right.timestamps[0]
    if not is_overlapping:
        if left.letters[-1] != " " and right.letters[0] != " ":
            one_ms_before_start = np.array([right.timestamps[0] - 0.001])
            new_suffix = TimestampedLetters(
                " " + right.letters,
                np.concatenate(one_ms_before_start, right.timestamps),
            )
        else:
            new_suffix = right

    else:
        left_cut, matches = cut_left_calc_matches(left, right, sm)
        if len(matches) > 0:
            glue_point_left_cut, glue_point_right = calc_glue_points(
                left_cut.letters, matches
            )
            # print(
            #     f"{left_cut.letters[:glue_point_left_cut]}---{right.letters[glue_point_right:]}"
            # )
            assert (
                left_cut.letters[glue_point_left_cut] == right.letters[glue_point_right]
            )
            new_suffix = TimestampedLetters(
                right.letters[(glue_point_right):],
                right.timestamps[(glue_point_right):],
            )
            time_diff = (
                left_cut.timestamps[glue_point_left_cut]
                - right.timestamps[glue_point_right]
            )
            new_suffix.timestamps[
                0
            ] += time_diff  # shift only the very first letter to exactly match the timestamp of the last letter in "left"
        else:
            new_suffix = NO_NEW_SUFFIX
            if DEBUG:
                print(f"not matches for: {left.letters} and {right.letters}")

    return new_suffix


@beartype
def calc_glue_points(
    left_cut_text: str, matches: list[difflib.Match]
) -> tuple[int, int]:
    aligned_idx = [(m.a + k, m.b + k) for m in matches for k in range(m.size)]
    dist_to_middle = [np.abs(i - round(len(left_cut_text) / 2)) for i, _ in aligned_idx]
    match_idx_closest_to_middle = np.argmin(dist_to_middle)
    glue_point_left, glue_point_right = aligned_idx[match_idx_closest_to_middle]
    return glue_point_left, glue_point_right


@beartype
def cut_left_calc_matches(
    left: TimestampedLetters,
    right: TimestampedLetters,
    sm: difflib.SequenceMatcher,
) -> tuple[TimestampedLetters, list[difflib.Match]]:
    """
    1. from left cut away what reaches too far into past
        left:________-:-
    2. from right cut away what reaches too far into future -> just for alignmend-method
        right:_______-:--
    3. find matches
    """
    tol = 0.5  # seconds:  for some reason I wanted half a second "space" to the left

    left_cut = left.slice(np.argwhere(left.timestamps > right.timestamps[0] - tol))
    assert len(left_cut.letters) > 0
    cut_right_just_to_help_alingment = right.slice(
        np.argwhere(right.timestamps < left.timestamps[-1])
    )

    assert len(cut_right_just_to_help_alingment.letters) > 0
    sm.set_seqs(left_cut.letters, cut_right_just_to_help_alingment.letters)
    matches = [m for m in sm.get_matching_blocks() if m.size > 0]
    return left_cut, matches

def accumulate_transcript_suffixes(
    suffixes_g: Iterable[TimestampedLetters],
) -> TimestampedLetters:
    prefix = None
    for suffix in suffixes_g:
        if prefix is not None:
            prefix = prefix.slice(np.argwhere(prefix.timestamps < suffix.timestamps[0]))
            prefix = TimestampedLetters(
                prefix.letters + suffix.letters,
                np.concatenate([prefix.timestamps, suffix.timestamps]),
            )
        else:
            prefix = suffix
    return prefix

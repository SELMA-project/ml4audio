import os
import sys

from beartype import beartype
from beartype.vale import Is

from misc_utils.beartypes import NeList
from misc_utils.utils import just_try
from ml4audio.audio_utils.aligned_transcript import (
    AlignedTranscript,
    LetterIdx,
)

sys.path.append(".")
import difflib
from typing import Tuple, Annotated

import numpy as np


DEBUG = os.environ.get("DEBUG_GLUER", "False").lower() != "false"
if DEBUG:
    print("DEBUGGING MODE for glue_left_right")

NonEmptyAlignedTranscript = Annotated[
    AlignedTranscript, Is[lambda x: x.len_not_space > 0]
]
# TODO: can this ever be empty?

"""
───▄▄▄
─▄▀░▄░▀▄
─█░█▄▀░█
─█░▀▄▄▀█▄█▄▀
▄▄█▄▄▄▄███▀

"""


@beartype
def glue_left_right_update_hyp_buffer(
    new_trans: NonEmptyAlignedTranscript,
    hyp_buffer: NonEmptyAlignedTranscript,
    sm: difflib.SequenceMatcher,
) -> tuple[str, str, NonEmptyAlignedTranscript]:

    # glueing may fail due to too few overlap
    glue_index, glued = just_try(
        lambda: glue_left_right(left=hyp_buffer, right=new_trans, sm=sm),
        default=(
            len(hyp_buffer),
            hyp_buffer,
        ),  # a failed glue does not add anything! In the hope that overlap is big enough so that it can be recovered by next glue!
        verbose=DEBUG,
        print_stacktrace=False,
    )
    assert len(glued.letters) > 0  # , f"{hyp_buffer.text=}, {new_trans.text=}"
    assert glued.letters[0].r_idx == 0
    ending_to_be_appended = glued.text[(glue_index):]
    ending_to_be_removed = hyp_buffer.text[(glue_index):]
    return ending_to_be_removed, ending_to_be_appended, glued


@beartype
def glue_left_right(
    left: NonEmptyAlignedTranscript,
    right: NonEmptyAlignedTranscript,
    sm: difflib.SequenceMatcher,
) -> tuple[int, NonEmptyAlignedTranscript]:
    """
    two overlapping sequences

    left:_____----:-
    right:_______-:----

    colon is glue-point which is to be found

    1. from left cut away what reaches too far into past
        left:________-:-
    2. from right cut away what reaches too far into future -> just for alignmend-method
        right:_______-:--

    return cut left and full right glued together
    """
    sr = right.sample_rate

    glue_index, letters_right = left_right_letters(left, right, sm)

    index_difference = right.offset - left.offset
    letters_right = update_letter_index(letters_right, index_difference)

    if DEBUG:
        print(
            f"GLUED left: {AlignedTranscript(left.letters[:(glue_index)], sr).text}, right: {AlignedTranscript(letters_right, sr).text}"
        )
    glued = AlignedTranscript(
        left.letters[:(glue_index)] + letters_right, sr, offset=left.offset
    ).update_offset()
    return glue_index, glued


@beartype
def left_right_letters(
    left: NonEmptyAlignedTranscript,
    right: NonEmptyAlignedTranscript,
    sm: difflib.SequenceMatcher,
) -> tuple[int, NeList[LetterIdx]]:
    sr = right.sample_rate
    assert sr == left.sample_rate
    is_overlapping = left.abs_idx(left.letters[-1]) > right.abs_idx(right.letters[0])
    if not is_overlapping:
        letters_right = prepent_space_letter(right)
        glue_index = len(left)

    else:
        left_cut, matches = cut_left_calc_matches(left, right, sm, sr)
        if len(matches) > 0:
            glue_point_left_cut, glue_point_right = calc_glue_points(left_cut, matches)
            glue_index = (
                glue_point_left_cut + (len(left) - len(left_cut)) + 1
            )  # +1 for exclusive
            letters_right = right.letters[(glue_point_right + 1) :]
        else:
            if DEBUG:
                print(f"not glueing but simply appending")
            letters_right = prepent_space_letter(left, right)
            glue_index = len(left)

    return glue_index, letters_right


@beartype
def calc_glue_points(left_cut, matches):
    aligned_idx = [(m.a + k, m.b + k) for m in matches for k in range(m.size)]
    dist_to_middle = [np.abs(i - round(len(left_cut.text) / 2)) for i, _ in aligned_idx]
    match_idx_closest_to_middle = np.argmin(dist_to_middle)
    glue_point_left, glue_point_right = aligned_idx[match_idx_closest_to_middle]
    return glue_point_left, glue_point_right


@beartype
def prepent_space_letter(right: NonEmptyAlignedTranscript) -> list[LetterIdx]:
    space_letter = LetterIdx(" ", right.letters[0].r_idx)
    letters_right = [space_letter] + right.letters
    return letters_right


def update_letter_index(
    letters: list[LetterIdx], index_difference: int
) -> list[LetterIdx]:
    return list(
        map(
            lambda x: LetterIdx(x.letter, x.r_idx + index_difference),
            letters,
        )
    )


@beartype
def cut_left_calc_matches(
    left: AlignedTranscript,
    right: AlignedTranscript,
    sm: difflib.SequenceMatcher,
    sr: int,
) -> tuple[AlignedTranscript, list[difflib.Match]]:
    tol = round(0.5 * sr)  # for some reason I wanted half a second "space" to the left
    letter_after_start_of_right = [
        s for s in left.letters if left.abs_idx(s) >= right.offset - tol
    ]
    left_cut = AlignedTranscript(
        letters=letter_after_start_of_right,
        sample_rate=sr,
        offset=left.offset,
    )
    assert len(left_cut.letters) > 0
    cut_right_just_to_help_alingment = AlignedTranscript(
        [
            l
            for l in right.letters
            if right.abs_idx(l) < left_cut.abs_idx(left_cut.letters[-1])
        ],
        sr,
        right.offset,
    )
    assert len(cut_right_just_to_help_alingment.letters) > 0
    sm.set_seqs(left_cut.text, cut_right_just_to_help_alingment.text)
    matches = [m for m in sm.get_matching_blocks() if m.size > 0]
    return left_cut, matches


@beartype
def remove_and_append(text: str, suffix_to_be_removed: str, new_suffix: str) -> str:
    """
    should only use by client, internally should grow the output via aligned transcripts
    """

    go_k_in_past = len(suffix_to_be_removed)
    assert (
        text.endswith(suffix_to_be_removed) or len(suffix_to_be_removed) == 0
    ), f"{text=} is not ending with {suffix_to_be_removed=}"

    stable_transcript = text[:-go_k_in_past] if go_k_in_past > 0 else text

    text = f"{stable_transcript}{new_suffix}"
    return text

import os
import sys

from beartype import beartype
from beartype.vale import Is

from misc_utils.beartypes import NeList
from misc_utils.utils import just_try
from ml4audio.audio_utils.aligned_transcript import (
    AlignedTranscript,
    LetterIdx,
    NeAlignedTranscript,
)

sys.path.append(".")
import difflib
from typing import Tuple, Annotated

import numpy as np

# LETTER_BUFFER_LEN = 100

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
) -> Tuple[str, str, NonEmptyAlignedTranscript]:

    # if DEBUG:
    #     print(
    #         f"left: {hyp_buffer.text=}, {hyp_buffer.offset=}, right: {new_trans.text=}, {new_trans.offset=}"
    #     )

    # glueing may fail due to too few overlap
    ending_to_be_removed, glued = just_try(
        lambda: glue_left_right(left=hyp_buffer, right=new_trans, sm=sm),
        default=(
            "",
            hyp_buffer,
        ),  # a failed glue does not add anything! In the hope that overlap is big enough so that it can be recovered by next glue!
        verbose=DEBUG,
        print_stacktrace=False,
    )
    assert hyp_buffer.text.endswith(
        ending_to_be_removed
    ), f"{glued.text=} doe snot end with {ending_to_be_removed=}"
    assert len(glued.letters) > 0  # , f"{hyp_buffer.text=}, {new_trans.text=}"
    assert glued.letters[0].r_idx == 0
    # TODO: ring-buffer?
    # letters = glued.letters[-LETTER_BUFFER_LEN:]
    # curr_transcr = AlignedTranscript(
    #     letters, TARGET_SAMPLE_RATE, glued.offset
    # ).update_offset()
    stable_len = len(hyp_buffer) - len(ending_to_be_removed)
    ending_to_be_appended = glued.text[stable_len:]
    return ending_to_be_removed, ending_to_be_appended, glued


@beartype
def glue_left_right(
    left: NeAlignedTranscript,
    right: NeAlignedTranscript,
    sm: difflib.SequenceMatcher,
) -> tuple[str, NeAlignedTranscript]:
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

    letters_left, ending_to_be_removed, letters_right = left_right_letters(
        left, right, sm
    )

    index_difference = right.offset - left.offset
    letters_right = update_letter_index(letters_right, index_difference)

    if DEBUG:
        print(
            f"GLUED left: {AlignedTranscript(letters_left, sr).text}, right: {AlignedTranscript(letters_right, sr).text}"
        )
    glued = NeAlignedTranscript(
        letters_left + letters_right, sr, offset=left.offset
    ).update_offset()
    return ending_to_be_removed, glued


@beartype
def left_right_letters(
    left: NeAlignedTranscript, right: NeAlignedTranscript, sm: difflib.SequenceMatcher
) -> tuple[NeList[LetterIdx], str, NeList[LetterIdx]]:
    sr = right.sample_rate
    assert sr == left.sample_rate
    is_overlapping = left.abs_idx(left.letters[-1]) > right.abs_idx(right.letters[0])
    ending_to_be_removed = ""
    if not is_overlapping:
        # if DEBUG:
        #     print("NOTHING to GLUE!")
        letters_left, letters_right = not_glueing_but_simply_appending(left, right)

    else:
        left_cut, matches = cut_left_calc_matches(left, right, sm, sr)
        if len(matches) > 0:

            ending_to_be_removed, letters_left, letters_right = __do_glue(
                left, left_cut, matches, right
            )
        else:
            if DEBUG:
                print(f"not glueing but simply appending")
            letters_left, letters_right = not_glueing_but_simply_appending(left, right)

    return letters_left, ending_to_be_removed, letters_right


@beartype
def __do_glue(
    left: AlignedTranscript,
    left_cut: AlignedTranscript,
    matches: NeList[difflib.Match],
    right: AlignedTranscript,
) -> Tuple[str, NeList[LetterIdx], NeList[LetterIdx]]:
    glue_point_left, glue_point_right = calc_glue_points(left_cut, matches)
    letters_right = right.letters[(glue_point_right + 1) :]
    k_to_be_removed = len(left_cut) - glue_point_left - 1
    letters_left = left.letters[:-k_to_be_removed]
    letters_left_removed = left.letters[-k_to_be_removed:]
    ending_to_be_removed = "".join([l.letter for l in letters_left_removed])
    if DEBUG:
        print(f"{matches=}, {left.text=},{left_cut.text=},{right.text=}")
    if has_no_text_at_all(letters_left) or has_no_text_at_all(letters_right):
        raise AssertionError(f"{matches=}, {left.text=},{left_cut.text=},{right.text=}")
    return ending_to_be_removed, letters_left, letters_right


@beartype
def calc_glue_points(left_cut, matches):
    aligned_idx = [(m.a + k, m.b + k) for m in matches for k in range(m.size)]
    dist_to_middle = [np.abs(i - round(len(left_cut.text) / 2)) for i, _ in aligned_idx]
    match_idx_closest_to_middle = np.argmin(dist_to_middle)
    glue_point_left, glue_point_right = aligned_idx[match_idx_closest_to_middle]
    return glue_point_left, glue_point_right


def has_no_text_at_all(letters):
    return len("".join([l.letter for l in letters if l.letter != " "])) == 0


def not_glueing_but_simply_appending(left, right):
    letters_left = left.letters
    space_letter = LetterIdx(" ", right.letters[0].r_idx)
    letters_right = [space_letter] + right.letters
    return letters_left, letters_right


def update_letter_index(letters, index_difference):
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
    tol = round(0.5 * sr)
    left_cut = AlignedTranscript(
        letters=[s for s in left.letters if left.abs_idx(s) >= right.offset - tol],
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

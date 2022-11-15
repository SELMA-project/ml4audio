# pylint: skip-file
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Generator, Optional, Tuple, Iterator

from beartype import beartype

from misc_utils.beartypes import NeStr, NeList

logger = logging.getLogger(__name__)
verbose_level = 0


class EditType(Enum):
    SUB = 1
    INS = 2
    DEL = 3
    EQU = 4  # correct


@dataclass
class Alignment:
    ref: str
    hyp: str
    refi_from: int
    hypi_from: int
    refi_to: Optional[int]
    hypi_to: Optional[int]  # tilo: why?
    eps: str

    def __post_init__(self):
        assert self.refi_to - self.refi_from <= 1, f"{self=}"
        assert self.hypi_to - self.hypi_from <= 1, f"{self=}"

    @property
    def edit_type(self) -> EditType:
        return get_edit_type(self.ref, self.hyp, self.eps)

    def get_padded_edt(
        self, hyp_tokens: list[str], ref_tokens: list[str], token_sep: str
    ):
        hyp_len = len(token_sep.join(hyp_tokens[self.hypi_from : self.hypi_to]))
        ref_len = len(token_sep.join(ref_tokens[self.refi_from : self.refi_to]))
        edt = edt_to_symbol(self.edit_type)
        if self.edit_type is EditType.INS:
            padded = edt * hyp_len
        elif self.edit_type is EditType.DEL:
            padded = edt * ref_len
        elif self.edit_type is EditType.SUB:
            len_diff = max(hyp_len - len(self.ref), ref_len - len(self.hyp))
            if len_diff > 0:
                padded = edt * hyp_len + " " * len_diff
            else:
                padded = edt * hyp_len
        elif self.edit_type is EditType.EQU:
            padded = edt * ref_len
        else:
            raise AssertionError
        return padded

    def get_padded_ref(self, hyp_tokens: list[str], token_sep: str):
        hyp_len = len(token_sep.join(hyp_tokens[self.hypi_from : self.hypi_to]))
        padded = self.ref
        if self.edit_type is EditType.INS:
            padded = self.eps * hyp_len
        elif self.edit_type is EditType.SUB:
            len_diff = hyp_len - len(self.ref)
            if len_diff > 0:
                padded = self.ref + self.eps * len_diff

        return padded

    def get_padded_hyp(self, ref_tokens: list[str], token_sep: str):
        ref_len = len(token_sep.join(ref_tokens[self.refi_from : self.refi_to]))
        padded = self.hyp
        if self.edit_type is EditType.DEL:
            padded = self.eps * ref_len
        elif self.edit_type is EditType.SUB:
            len_diff = ref_len - len(self.hyp)
            if len_diff > 0:
                padded = self.hyp + self.eps * len_diff

        return padded


@beartype
def smith_waterman_alignment(
    ref,
    hyp,
    eps_symbol="|",
    similarity_score_function=lambda x, y: 2 if (x == y) else -1,
    del_score=-1,
    ins_score=-1,
    align_full_hyp=True,
) -> Tuple[list[Alignment], float]:
    """
    stolen from kaldi; see egs/wsj/s5/steps/cleanup/internal/align_ctm_ref.py

    Does Smith-Waterman alignment of reference sequence and hypothesis
    sequence.
    This is a special case of the Smith-Waterman alignment that assumes that
    the deletion and insertion costs are linear with number of incorrect words.

    If align_full_hyp is True, then the traceback of the alignment
    is started at the end of the hypothesis. This is when we want the
    reference that aligns with the full hypothesis.
    This differs from the normal Smith-Waterman alignment, where the traceback
    is from the highest score in the alignment score matrix. This
    can be obtained by setting align_full_hyp as False. This gets only the
    sub-sequence of the hypothesis that best matches with a
    sub-sequence of the reference.

    Returns a list of tuples where each tuple has the format:
        (ref_word, hyp_word, ref_word_from_index, hyp_word_from_index,
         ref_word_to_index, hyp_word_to_index)
    """
    output = []

    ref_len = len(ref)
    hyp_len = len(hyp)

    bp = [[] for x in range(ref_len + 1)]

    # Score matrix of size (ref_len + 1) x (hyp_len + 1)
    # The index m, n in this matrix corresponds to the score
    # of the best matching sub-sequence pair between reference and hypothesis
    # ending with the reference word ref[m-1] and hypothesis word hyp[n-1].
    # If align_full_hyp is True, then the hypothesis sub-sequence is from
    # the 0th word i.e. hyp[0].
    H = [[] for x in range(ref_len + 1)]

    for ref_index in range(ref_len + 1):
        if align_full_hyp:
            H[ref_index] = [-(hyp_len + 2) for x in range(hyp_len + 1)]
            H[ref_index][0] = 0
        else:
            H[ref_index] = [0 for x in range(hyp_len + 1)]
        bp[ref_index] = [(0, 0) for x in range(hyp_len + 1)]

        if align_full_hyp and ref_index == 0:
            for hyp_index in range(1, hyp_len + 1):
                H[0][hyp_index] = H[0][hyp_index - 1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index - 1)
                logger.debug(
                    "({},{}) -> ({},{}): {}"
                    "".format(
                        ref_index,
                        hyp_index - 1,
                        ref_index,
                        hyp_index,
                        H[ref_index][hyp_index],
                    )
                )

    max_score = -float("inf")
    max_score_element = (0, 0)

    for ref_index in range(1, ref_len + 1):  # Reference
        for hyp_index in range(1, hyp_len + 1):  # Hypothesis
            sub_or_ok = H[ref_index - 1][hyp_index - 1] + similarity_score_function(
                ref[ref_index - 1], hyp[hyp_index - 1]
            )

            if (not align_full_hyp and sub_or_ok > 0) or (
                align_full_hyp and sub_or_ok >= H[ref_index][hyp_index]
            ):
                H[ref_index][hyp_index] = sub_or_ok
                bp[ref_index][hyp_index] = (ref_index - 1, hyp_index - 1)
                logger.debug(
                    "({},{}) -> ({},{}): {} ({},{})"
                    "".format(
                        ref_index - 1,
                        hyp_index - 1,
                        ref_index,
                        hyp_index,
                        H[ref_index][hyp_index],
                        ref[ref_index - 1],
                        hyp[hyp_index - 1],
                    )
                )

            if H[ref_index - 1][hyp_index] + del_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index - 1][hyp_index] + del_score
                bp[ref_index][hyp_index] = (ref_index - 1, hyp_index)
                logger.debug(
                    "({},{}) -> ({},{}): {}"
                    "".format(
                        ref_index - 1,
                        hyp_index,
                        ref_index,
                        hyp_index,
                        H[ref_index][hyp_index],
                    )
                )

            if H[ref_index][hyp_index - 1] + ins_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index][hyp_index - 1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index - 1)
                logger.debug(
                    "({},{}) -> ({},{}): {}"
                    "".format(
                        ref_index,
                        hyp_index - 1,
                        ref_index,
                        hyp_index,
                        H[ref_index][hyp_index],
                    )
                )

            # if hyp_index == hyp_len and H[ref_index][hyp_index] >= max_score:
            if (not align_full_hyp or hyp_index == hyp_len) and H[ref_index][
                hyp_index
            ] >= max_score:
                max_score = H[ref_index][hyp_index]
                max_score_element = (ref_index, hyp_index)

    ref_index, hyp_index = max_score_element
    score = max_score
    logger.debug("Alignment score: %s for (%d, %d)", score, ref_index, hyp_index)

    while (not align_full_hyp and score >= 0) or (align_full_hyp and hyp_index > 0):
        try:
            prev_ref_index, prev_hyp_index = bp[ref_index][hyp_index]
            if (prev_ref_index, prev_hyp_index) == (ref_index, hyp_index) or (
                prev_ref_index,
                prev_hyp_index,
            ) == (0, 0):
                score = H[ref_index][hyp_index]
                if score != 0:
                    ref_word = ref[ref_index - 1] if ref_index > 0 else eps_symbol
                    hyp_word = hyp[hyp_index - 1] if hyp_index > 0 else eps_symbol
                    output.append(
                        (
                            ref_word,
                            hyp_word,
                            prev_ref_index,
                            prev_hyp_index,
                            ref_index,
                            hyp_index,
                        )
                    )

                    ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
                    score = H[ref_index][hyp_index]
                break

            if ref_index == prev_ref_index + 1 and hyp_index == prev_hyp_index + 1:
                # Substitution or correct
                output.append(
                    (
                        ref[ref_index - 1] if ref_index > 0 else eps_symbol,
                        hyp[hyp_index - 1] if hyp_index > 0 else eps_symbol,
                        prev_ref_index,
                        prev_hyp_index,
                        ref_index,
                        hyp_index,
                    )
                )
            elif prev_hyp_index == hyp_index:
                # Deletion
                assert prev_ref_index == ref_index - 1
                output.append(
                    (
                        ref[ref_index - 1] if ref_index > 0 else eps_symbol,
                        eps_symbol,
                        prev_ref_index,
                        prev_hyp_index,
                        ref_index,
                        hyp_index,
                    )
                )
            elif prev_ref_index == ref_index:
                # Insertion
                assert prev_hyp_index == hyp_index - 1
                output.append(
                    (
                        eps_symbol,
                        hyp[hyp_index - 1] if hyp_index > 0 else eps_symbol,
                        prev_ref_index,
                        prev_hyp_index,
                        ref_index,
                        hyp_index,
                    )
                )
            else:
                raise RuntimeError

            ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
            score = H[ref_index][hyp_index]
        except Exception:
            logger.error(
                "Unexpected entry (%d,%d) -> (%d,%d), %s, %s",
                prev_ref_index,
                prev_hyp_index,
                ref_index,
                hyp_index,
                ref[prev_ref_index],
                hyp[prev_hyp_index],
            )
            raise RuntimeError("Unexpected result: Bug in code!!")

    assert align_full_hyp or score == 0

    output.reverse()

    if verbose_level > 2:
        for ref_index in range(ref_len + 1):
            for hyp_index in range(hyp_len + 1):
                print(f"{H[ref_index][hyp_index]} ", end="", file=sys.stderr)
            print("", file=sys.stderr)

    logger.debug("Aligned output:")
    logger.debug("  -  ".join([f"({x[4]},{x[5]})" for x in output]))
    logger.debug("REF: ")
    logger.debug("    ".join(str(x[0]) for x in output))
    logger.debug("HYP:")
    logger.debug("    ".join(str(x[1]) for x in output))

    return [Alignment(*o, eps=eps_symbol) for o in output], float(max_score)


@beartype
def get_edit_type(ref: str, hyp: str, eps="|") -> EditType:
    if ref != hyp and not (ref == eps or hyp == eps):
        et = EditType.SUB
    elif ref != hyp and ref == eps:
        et = EditType.INS
    elif ref != hyp and hyp == eps:
        et = EditType.DEL
    else:
        et = EditType.EQU  # correct, so actuall not edit at all

    return et


@beartype
def padded_smith_waterman_alignments(
    ref_tok: NeList[str], hyp_tok: NeList[str], eps="|"
) -> NeList[Alignment]:
    alignments, score = smith_waterman_alignment(
        ref_tok,
        hyp_tok,
        similarity_score_function=lambda x, y: 2 if (x == y) else -1,
        del_score=-1,
        ins_score=-1,
        eps_symbol=eps,
        align_full_hyp=True,
    )
    start = alignments[0].refi_from
    deletions_left = [
        Alignment(ref_tok[i], eps, i, 0, refi_to=i + 1, hypi_to=0 + 1, eps=eps)
        for i in range(0, start)
    ]
    # for x in alignments:
    #     x.hypi_from+=start
    #     x.hypi_to+=start

    end = alignments[-1].refi_from
    deletions_right = [
        Alignment(
            ref_tok[i],
            eps,
            i,
            len(hyp_tok),
            refi_to=i + 1,
            hypi_to=len(hyp_tok) + 1,
            eps=eps,
        )
        for i in range(end + 1, len(ref_tok))
    ]
    alignments = deletions_left + alignments + deletions_right
    return alignments


def calc_aligned_ngram_tuples(
    ref_tok: list[str], hyp_tok: list[str], order: int
) -> Generator[Tuple[list[str], list[str]], None, None]:
    alignments = padded_smith_waterman_alignments(ref_tok, hyp_tok)

    ri_to_alignment = defaultdict(list)
    for a in alignments:
        ri_to_alignment[a.refi_from].append(a)

    assert all([len(v) > 0 for v in ri_to_alignment.values()])

    for o in range(1, order + 1):
        for k in range(len(ref_tok) - (o - 1)):
            ngram = [al for i in range(k, k + o) for al in ri_to_alignment[i]]
            ref_ngram = ref_tok[ngram[0].refi_from : (ngram[-1].refi_from + 1)]
            hyp_ngram = hyp_tok[ngram[0].hypi_from : (ngram[-1].hypi_from + 1)]
            yield (hyp_ngram, ref_ngram)


@beartype
def edt_to_symbol(edt: EditType) -> str:
    if edt == EditType.SUB:
        s = "c"
    elif edt == EditType.INS:
        s = "+"
    elif edt == EditType.DEL:
        s = "-"
    else:  # edt==EditType.COR:
        s = " "
    return s


@beartype
def regex_tokenizer(
    text: str, pattern=r"\w+(?:'\w+)?|[^\w\s]"
) -> list[Tuple[int, int, str]]:  # pattern stolen from scikit-learn
    return [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, text)]


@beartype
def calc_error_word_tuples(ref: NeStr, hyp: NeStr, eps: str) -> list[Tuple[str, str]]:
    alignments = padded_smith_waterman_alignments(list(ref), list(hyp), eps)
    padded_ref = "".join(x.ref for x in alignments)
    padded_hyp = "".join(x.hyp for x in alignments)
    ref_tokens = regex_tokenizer(padded_ref, pattern=r"\w+(?:'\w+)?|[^\w\s]")
    word_tuples = [
        (t.replace(eps, ""), padded_hyp[s:e].replace(eps, "")) for s, e, t in ref_tokens
    ]
    error_word_tuples = [(r, h) for r, h in word_tuples if h != r]
    return error_word_tuples


@beartype
def align_split(
    a: str, b: str, split_len_a: int = 50, debug=False
) -> Tuple[list[str], list[str]]:
    """
    useful to help stupid difflib (which is used in icdiff) to handle pairs with more severe differences
    splits at first space after split_len_a
    """
    eps = "_"
    alignments = padded_smith_waterman_alignments(list(a), list(b), eps)

    if debug:
        edts: list[EditType] = [get_edit_type(a.ref, a.hyp, eps) for a in alignments]
        padded_ref = "".join(x.ref for x in alignments)
        print("ref: " + padded_ref)
        padded_hyp = "".join(x.hyp for x in alignments)
        print("hyp: " + padded_hyp)
        print(f"edt: {''.join([edt_to_symbol(e) for e in edts])}")

    def split_into_lines():
        buffer_a = ""
        buffer_b = ""
        for al in alignments:
            if len(buffer_a) > split_len_a and buffer_a[-1] == " ":
                yield buffer_a, buffer_b
                buffer_a, buffer_b = "", ""

            buffer_a += a[al.refi_from : al.refi_to]
            buffer_b += b[al.hypi_from : al.hypi_to]
        yield buffer_a, buffer_b

    refs_hyps = list(split_into_lines())
    a_s, b_s = (list(x) for x in zip(*refs_hyps))
    return a_s, b_s


@beartype
def align_group(a: str, b: str, max_num_chars=60, eps="_") -> list[list[Alignment]]:
    """
    groups as many alignments together as fit into upper limit of max_num_chars
    to fit on one line
    """
    alignments = padded_smith_waterman_alignments(list(a), list(b), eps)

    def generate_aligned_pairs() -> Iterator[list[Alignment]]:
        buffer = []
        for al in alignments:
            new_ref_len = sum((len(a.ref) for a in buffer)) + len(al.ref)
            new_hyp_len = sum((len(a.hyp) for a in buffer)) + len(al.hyp)
            if max(new_ref_len, new_hyp_len) < max_num_chars:
                buffer.append(al)
            else:
                yield buffer
                buffer = [al]

        yield buffer

    return list(generate_aligned_pairs())


if __name__ == "__main__":
    mode = "char"

    ref = "I'd think the cat is black"
    hyp = "d hee cad i blac"

    ref = "im jahr neunzehn hundert zwölf bis ende"
    hyp = "im ja 19 100 zwölf bis ende"
    #
    # ref = "eEe d e dd EeE eeee dd"
    # hyp = "Ii eEe e EeE iii eeee"

    verbose = 3
    eps = "_"  # "…"
    #
    # output, score = smith_waterman_alignment(
    #     ref,
    #     hyp,
    #     similarity_score_function=lambda x, y: 2 if (x == y) else -1,
    #     del_score=-1,
    #     ins_score=-1,
    #     eps_symbol=eps,
    #     align_full_hyp=True,
    # )
    #
    # print("ref: " + "".join(x.ref for x in output))
    # print("hyp: " + "".join(x.hyp for x in output))

    name2tokenize_fun = {
        "space": (lambda s: s.split(" "), " "),
        "char": (lambda s: list(s), ""),
    }
    tokenize_fun, token_sep = name2tokenize_fun[mode]
    hyp_tok = tokenize_fun(hyp)
    ref_tok = tokenize_fun(ref)
    alignments = padded_smith_waterman_alignments(ref_tok, hyp_tok, eps)
    pprint(alignments)
    edts: list[str] = [
        a.get_padded_edt(hyp_tok, ref_tok, token_sep) for a in alignments
    ]

    print("padded")
    padded_ref = token_sep.join(
        x.get_padded_ref(hyp_tok, token_sep) for x in alignments
    )
    print("ref_pad: " + padded_ref)
    padded_hyp = token_sep.join(
        x.get_padded_hyp(ref_tok, token_sep) for x in alignments
    )
    print("hyp_pad: " + padded_hyp)
    print(f"edt_pad: {token_sep.join(edts)}")

    error_word_tuples = calc_error_word_tuples(ref, hyp, eps)
    print(f"{error_word_tuples=}")

    def striken(text):
        return "\u0336" + "\u0336".join(text)

    # print(striken("this is a test"))
    # pprint([f"ref: {r}, hyp: {h}" for h,r in calc_aligned_ngram_tuples(list(ref), list(hyp), 3) if h != r and len(r) == 3])

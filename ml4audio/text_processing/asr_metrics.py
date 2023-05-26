from typing import Tuple

import Levenshtein
import jiwer as jiwer
from beartype import beartype

from misc_utils.beartypes import NeList

"""
nemo is using [editdistance](https://github.com/roy-ht/editdistance)

here Levenshtein is used: https://github.com/ztane/python-Levenshtein/
"""


@beartype
def calc_num_word_errors(hyp: str, ref: str) -> Tuple[int, int]:
    """
    based on: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/decoder.py
    https://github.com/SeanNaren/deepspeech.pytorch/blob/78f7fb791f42c44c8a46f10e79adad796399892b/deepspeech_pytorch/decoder.py#L42
    """

    def tokenize(s):
        return s.split()

    b = set(tokenize(hyp) + tokenize(ref))
    token2idx = {t: k for k, t in enumerate(b)}

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(token2idx[w]) for w in hyp.split()]
    w2 = [chr(token2idx[w]) for w in ref.split()]

    len_ref = len(ref.split(" "))
    return Levenshtein.distance("".join(w1), "".join(w2)), len_ref


@beartype
def calc_num_char_erros(hyp: str, ref: str) -> Tuple[int, int]:
    """
    based on: https://github.com/SeanNaren/deepspeech.pytorch/blob/78f7fb791f42c44c8a46f10e79adad796399892b/deepspeech_pytorch/decoder.py#L62
    """
    hyp, ref, = (
        hyp.replace(" ", ""),
        ref.replace(" ", ""),
    )  # TODO(tilo): why removing spaces?

    len_ref = len(ref.replace(" ", ""))
    return Levenshtein.distance(hyp, ref), len_ref


@beartype
def micro_average(errors_lens: NeList[tuple[int, int]]) -> float:
    num_tokens = sum([l for _, l in errors_lens])
    errors = sum([s for s, _ in errors_lens])
    return float(errors) / float(num_tokens) if num_tokens > 0 else -1.0


@beartype
def micro_avg_wer(hyps_targets: NeList[Tuple[str, str]]):
    errors_lens = [calc_num_word_errors(hyp, ref) for hyp, ref in hyps_targets]
    return micro_average(errors_lens)


@beartype
def micro_avg_cer(hyps_targets: NeList[Tuple[str, str]]):
    errors_lens = [calc_num_char_erros(hyp, ref) for hyp, ref in hyps_targets]
    return micro_average(errors_lens)


calc_cer = micro_avg_cer  # old deprecated name
calc_wer = micro_avg_wer  # old deprecated name


@beartype
def micro_avg_asr_scores(
    id_hyp_target: NeList[tuple[str, str, str]]
) -> dict[str, float]:
    hyp_targets = [(h, t) for _, h, t in id_hyp_target]
    # editops=[Levenshtein.editops(h, t) for h,t in hyp_targets]
    hyps, targets = [list(x) for x in zip(*hyp_targets)]
    num_chars = sum(len(t) for t in targets)
    cho = jiwer.process_characters(targets, hyps)
    return {
        "wer": micro_avg_wer(hyp_targets),
        "cer": micro_avg_cer(hyp_targets),
        "ins": cho.insertions / num_chars,
        "del": cho.deletions / num_chars,
        "sub": cho.substitutions / num_chars,
        "hit": cho.hits / num_chars,
    }


calc_asr_scores = micro_avg_asr_scores  # old deprecated name

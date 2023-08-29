import jiwer as jiwer
from beartype import beartype

from misc_utils.beartypes import NeList, NeStr


@beartype
def character_error_rates(refs: NeList[str], hyps: NeList[str]) -> dict[str, float]:
    cho = jiwer.process_characters(refs, hyps)
    num_chars = sum([len(r) for r in cho.references])
    return {
        "cer": cho.cer,
        "insr": cho.insertions / num_chars,
        "delr": cho.deletions / num_chars,
        "subr": cho.substitutions / num_chars,
        # "hit": cho.hits / num_chars, # who cares about a hit-rate?
    }


@beartype
def word_error_rates(refs: NeList[str], hyps: NeList[str]) -> dict[str, float]:
    who = jiwer.process_words(refs, hyps)
    num_words = sum(len(r) for r in who.references)
    assert num_words == who.hits + who.deletions + who.substitutions
    return {
        "wer": who.wer,
        "insr": who.insertions / num_words,
        "delr": who.deletions / num_words,
        "subr": who.substitutions / num_words,
        # "hit": who.hits / num_words,
    }


@beartype
def calc_cer(refs: NeList[NeStr], hyps: NeList[str]) -> float:
    return character_error_rates(refs, hyps)["cer"]


@beartype
def calc_wer(refs: NeList[NeStr], hyps: NeList[str]) -> float:
    return word_error_rates(refs, hyps)["wer"]


@beartype
def micro_avg_asr_scores(
    refs_hyps: NeList[tuple[NeStr, str]]
) -> dict[str, dict[str, float]]:
    refs, hyps = [list(x) for x in zip(*refs_hyps)]
    return {
        "word": word_error_rates(refs, hyps),
        "char": character_error_rates(refs, hyps),
    }

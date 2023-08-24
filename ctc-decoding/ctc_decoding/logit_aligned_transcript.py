from dataclasses import dataclass
from typing import Optional

from beartype import beartype

from misc_utils.beartypes import NeStr, NeList

TokenSpans = list[tuple[str, tuple[int, int]]]


@dataclass
class LogitAlignedTranscript:
    """
    Text is character-wise aligned to logits, no time-stamps here.
        logits == ctc-matrix
    """

    text: NeStr
    logit_ids: NeList[int]  # TODO: not too strict?

    logits_score: Optional[float] = None
    lm_score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate data."""
        have_same_len = len(self.text) == len(self.logit_ids)
        assert have_same_len, (
            f"{self.text=} and {self.logit_ids=} have different length! "
            + f"{len(self.text)=}!={len(self.logit_ids)=}"
        )

    @staticmethod
    def create_from_token_spans(
        token_spans: TokenSpans,
        lm_score: float,
        logits_score: float,
    ):
        text = " ".join([tok for tok, _ in token_spans])
        return LogitAlignedTranscript(
            text=text,
            logit_ids=charwise_idx_for_tokenspans_via_linear_interpolation(token_spans),
            lm_score=lm_score,
            logits_score=logits_score,
        )


@beartype
def charwise_idx_for_tokenspans_via_linear_interpolation(
    token_spans: TokenSpans,
) -> list[int]:
    seq_idx = [
        round(start + (end - start) * k / len(word))  # interpolate
        for word, (start, end) in token_spans
        for k in range(len(word) + 1)
    ]
    return seq_idx[:-1]  # all but the last one, which is a space

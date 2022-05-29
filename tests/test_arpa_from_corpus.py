import tempfile

from misc_utils.buildable import BuildableList
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.text_processing.asr_text_normalization import TranscriptNormalizer, Casing
from ml4audio.text_processing.kenlm_arpa import ArpaArgs, ArpaBuilder
from ml4audio.text_processing.word_based_text_corpus import (
    WordBasedLMCorpus,
    RglobRawCorpus,
)
from conftest import TEST_RESOURCES


def test_arpa_from_corpus(vocab):
    test_corpus_dir = TEST_RESOURCES
    normalizer = TranscriptNormalizer(
        casing=Casing.upper, text_normalizer="en", vocab=vocab
    )

    with tempfile.TemporaryDirectory() as cache_base:
        BASE_PATHES["tmp"] = cache_base
        cache_base = PrefixSuffix("tmp", "")

        arpa_args = ArpaArgs(
            order=5,
            prune="|".join(str(k) for k in [0, 8, 16]),
        )

        lm_corpus = WordBasedLMCorpus(
            name="test",
            cache_base=cache_base,
            raw_corpora=BuildableList[RglobRawCorpus](
                [
                    RglobRawCorpus(
                        cache_base=cache_base,
                        corpus_dir=test_corpus_dir,
                        file_pattern="*corpus.txt",
                    )
                ]
            ),
            normalizer=normalizer,
        )

        arpa_builder = ArpaBuilder(
            cache_base=cache_base,
            arpa_args=arpa_args,
            corpus=lm_corpus,
        )

        arpa_builder.build()
        # TODO: not sure how to check arpa-file for validity
        # shutil.copy(arpa_file,f"{test_corpus_dir}/{Path(arpa_file).name}")

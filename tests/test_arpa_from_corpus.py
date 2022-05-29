import tempfile

from data_io.readwrite_files import read_lines
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
        some_lines=list(read_lines(arpa_builder.arpa_filepath,limit=20))
        # fmt: off
        expected_lines=['', '\\data\\', 'ngram 1=669', 'ngram 2=2', 'ngram 3=0', 'ngram 4=0', 'ngram 5=0', '', '\\1-grams:', '-3.1839507\t<unk>\t0', '0\t<s>\t-0.03075325', '-1.26055\t</s>\t0', '-1.8272647\tA\t0', '-2.8700492\tMYTH\t0', '-1.8468318\tIS\t0', '-3.0925605\tFANCIFUL\t0', '-3.0925605\tEXPLANATION\t0', '-1.4614682\tOF\t-0.07636787', '-2.8700492\tGIVEN\t0', '-3.0925605\tPHENOMENON\t0']
        # fmt: on
        assert expected_lines==some_lines  # TODO: not sure how to check arpa-file for validity

import pytest

from ml4audio.text_processing.character_mappings.text_normalization import (
    CHARACTER_MAPPINGS,
)


@pytest.mark.parametrize(
    "lang_input_output",
    [
        ("en", "Jon-Do–e's", "Jon-Do-e's"),
        ("en", 'Jon-Do–e"s', "Jon-Do-e s"),
        ("de", "Jon-Do–e's", "Jon-Do-e s"),
        ("ru", "this is a test", "дис ис а тест"),
        ("ru", "dwaja mat, pisdez", "дwая мат писдец"),
    ],
)
def test_text_normalization(lang_input_output):
    lang, input, output = lang_input_output
    assert CHARACTER_MAPPINGS[lang](input) == output

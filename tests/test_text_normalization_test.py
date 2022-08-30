import pytest

from ml4audio.text_processing.character_mappings.character_mapping import (
    CHARACTER_MAPPINGS,
)


@pytest.mark.parametrize(
    "lang_input_output",
    [
        ("en", "Jon-Do–e's", "Jon-Do-e's"),
        ("en", 'Jon-Do–e"s', "Jon-Do-e s"),
        ("de", "Jon-Do–e's", "Jon-Do-e s"),
    ],
)
def test_text_normalization(lang_input_output):
    lang, input, output = lang_input_output
    assert CHARACTER_MAPPINGS[lang](input) == output

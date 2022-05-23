import abc
import re
import string

from ml4audio.text_processing.character_mappings.character_maps import (
    ENGLISH_CHARACTER_MAPPING,
    PUNCTUATION_MAPPING,
)
from ml4audio.text_processing.character_mappings.not_str_translatable_maps import (
    SAME_SAME_BUT_DIFFERENT,
)


class PluginNameConflictError(BaseException):
    """more than 1 plugin of same name"""


def register_normalizer_plugin(name):
    if name in CHARACTER_MAPPINGS:
        raise PluginNameConflictError(
            f"you have more than one TextNormalizer of name {name}"
        )

    def register_wrapper(clazz):
        plugin = clazz()
        CHARACTER_MAPPINGS[name] = plugin

    return register_wrapper


class TextNormalizer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, text: str) -> str:
        pass


class CharacterMapping(TextNormalizer):
    @property
    @abc.abstractmethod
    def mapping(self) -> dict[str, str]:
        pass

    def __init__(self) -> None:
        # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
        self.table = str.maketrans(self.mapping)

    def __call__(self, text: str) -> str:
        text = text.translate(self.table)
        for k, v in SAME_SAME_BUT_DIFFERENT.items():
            text = text.replace(k, v)
        text = re.sub(r"\s+", " ", text)
        return text


CHARACTER_MAPPINGS: dict[str, CharacterMapping] = {}


@register_normalizer_plugin("none")
class NoCharacterMappingAtAll(CharacterMapping):
    @property
    def mapping(self) -> dict[str, str]:
        return {}


@register_normalizer_plugin("no_punct")
class NoPunctuation(CharacterMapping):
    @property
    def mapping(self) -> dict[str, str]:
        PUNCTUATION = string.punctuation + "„“’”'-—…"
        PUNCTUATION_TO_BE_REPLACE_BY_SPACE = {key: " " for key in PUNCTUATION}
        return PUNCTUATION_TO_BE_REPLACE_BY_SPACE


@register_normalizer_plugin("de")
class GermanTextNormalizer(CharacterMapping):
    @property
    def mapping(self) -> dict[str, str]:
        GERMAN_WHITE_LIST = {"ä", "ü", "ö", "ß"}

        GERMAN_CHARACTER_MAPPING = {
            k: v
            for k, v in ENGLISH_CHARACTER_MAPPING.items()
            if k not in GERMAN_WHITE_LIST
        }

        return {**GERMAN_CHARACTER_MAPPING, **PUNCTUATION_MAPPING}


@register_normalizer_plugin("es")
class SpanishTextNormalizer(CharacterMapping):
    @property
    def mapping(self) -> dict[str, str]:
        SPANISH_WHITE_LIST = {"ñ", "ü", "ö", "á", "é", "í", "ó", "ú"}
        SPANISH_CHARACTER_MAPPING = {
            k: v
            for k, v in ENGLISH_CHARACTER_MAPPING.items()
            if k not in SPANISH_WHITE_LIST
        }
        return {**SPANISH_CHARACTER_MAPPING, **PUNCTUATION_MAPPING}


@register_normalizer_plugin("en")
class EnglishTextNormalizer(CharacterMapping):
    @property
    def mapping(self) -> dict[str, str]:
        return {**ENGLISH_CHARACTER_MAPPING, **PUNCTUATION_MAPPING}

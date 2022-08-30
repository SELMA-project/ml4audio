import abc
import re
import string

from unicode_tr import unicode_tr

from ml4audio.text_processing.character_mappings.cyrillic_character_maps import (
    NO_JO,
    RECOVER_CYRILLIC,
)
from ml4audio.text_processing.character_mappings.latin_character_maps import (
    REMOVE_EVERYTHING,
    REPLACE_ALL_PUNCT_WITH_SPACE,
    NORMALIZE_APOSTROPHES,
    NORMALIZE_DASH,
)
from ml4audio.text_processing.character_mappings.not_str_translatable_maps import (
    SAME_SAME_BUT_DIFFERENT,
)


class PluginNameConflictError(BaseException):
    """more than 1 plugin of same name"""


def register_normalizer_plugin(name):
    """
    TODO: why not simple single-ton instead?
    all these "plugins" get instantiated during import-time! is this really what I want?
    """
    if name in CHARACTER_MAPPINGS:
        raise PluginNameConflictError(
            f"you have more than one TextNormalizer of name {name}"
        )

    def register_wrapper(clazz):
        plugin = clazz()
        CHARACTER_MAPPINGS[name] = plugin

    return register_wrapper


class TextNormalizer(abc.ABC):
    # TODO: rename to TextCleaner ?
    @abc.abstractmethod
    def __call__(self, text: str) -> str:
        pass


TextCleaner = TextNormalizer


class CharacterMapping(TextNormalizer):
    @property
    @abc.abstractmethod
    def mapping(self) -> dict[str, str]:
        pass

    def __init__(self) -> None:
        # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
        self.table = str.maketrans(self.mapping)

    @property
    def replace_mapping(self):
        return SAME_SAME_BUT_DIFFERENT

    def __call__(self, text: str) -> str:
        for k, v in self.replace_mapping.items():
            text = text.replace(k, v)
        text = text.translate(self.table)
        text = re.sub(r"\s+", " ", text)
        return text


CHARACTER_MAPPINGS: dict[str, CharacterMapping] = {}
TEXT_CLEANERS: dict[str, TextCleaner] = CHARACTER_MAPPINGS  # TODO: use this in future?


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
        white_list = {"ä", "ü", "ö", "ß"}
        return {
            k: v
            for k, v in (
                REMOVE_EVERYTHING | REPLACE_ALL_PUNCT_WITH_SPACE | NORMALIZE_DASH
            ).items()
            if k not in white_list
        }


@register_normalizer_plugin("ru")
class RussianTextNormalizer(CharacterMapping):
    @property
    def replace_mapping(self):
        multiletter = {
            "ch": "ч",
            "sh": "ш",  # cannot map multi-letter here
            # "sh": "щ",
            "you": "ю",
            "ja": "я",
            "th": "д",
        }
        return SAME_SAME_BUT_DIFFERENT | multiletter

    @property
    def mapping(self) -> dict[str, str]:
        white_list = {}
        return {
            k: v
            for k, v in (
                REPLACE_ALL_PUNCT_WITH_SPACE | RECOVER_CYRILLIC | NO_JO
            ).items()
            if k not in white_list
        }


@register_normalizer_plugin("tr")
class TurkishTextCleaner(TextCleaner):
    @property
    def mapping(self) -> dict[str, str]:
        turkish_white_list = ["ç", "ö", "ü", "ğ", "ı", "ş"]

        return {
            k: v for k, v in REMOVE_EVERYTHING.items() if k not in turkish_white_list
        } | REPLACE_ALL_PUNCT_WITH_SPACE

    def __init__(self) -> None:
        # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
        self.table = str.maketrans(self.mapping)

    def __call__(self, text: str) -> str:
        text = text.translate(self.table)
        for k, v in SAME_SAME_BUT_DIFFERENT.items():
            text = text.replace(k, v)
        text = re.sub(r"\s+", " ", text)
        instance_of_subclass_of_str_overring_upper_lower_methods = unicode_tr(text)
        return instance_of_subclass_of_str_overring_upper_lower_methods


@register_normalizer_plugin("es")
class SpanishTextNormalizer(CharacterMapping):
    @property
    def mapping(self) -> dict[str, str]:
        SPANISH_WHITE_LIST = {"ñ", "ü", "ö", "á", "é", "í", "ó", "ú"}
        SPANISH_CHARACTER_MAPPING = {
            k: v for k, v in REMOVE_EVERYTHING.items() if k not in SPANISH_WHITE_LIST
        }
        return {**SPANISH_CHARACTER_MAPPING, **REPLACE_ALL_PUNCT_WITH_SPACE}


@register_normalizer_plugin("en")
class EnglishTextNormalizer(CharacterMapping):
    @property
    def mapping(self) -> dict[str, str]:
        english_white_list = ["-", "'"]
        return {
            k: v
            for k, v in (
                REMOVE_EVERYTHING
                | REPLACE_ALL_PUNCT_WITH_SPACE
                | NORMALIZE_APOSTROPHES
                | NORMALIZE_DASH
            ).items()
            if k not in english_white_list
        }

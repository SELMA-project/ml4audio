"""
see: https://www.businessballs.com/glossaries-and-terminology/accents-and-diacritical-marks/
é - accent acute
è - accent grave
ê - circumflex
ë - umlaut or diaerisis
ç - cedilla
ñ - tilde
ø - streg
ð - eth (capital form Ð)
å - bolle
æ - ligature
œ - ligature
ē - macron
č - háček
ŭ - crescent

TODO: what about upper-cased letters!!
"""

# assuming that this backward accent is just typo
import string

remove_backward_accent = {
    "à": "a",
    "è": "e",
    "ì": "i",
    "ò": "o",
    "ù": "u",
}

# hats, circumflex
remove_hats = {
    "â": "a",
    "ê": "e",
    "ô": "o",
    "î": "i",
    "û": "u",
}

remove_tilde = {
    "ã": "a",
    "ñ": "n",
}
remove_flat = {
    "ō": "o",
    "ē": "e",
}


remove_accent = {
    # accent acute
    "ń": "n",
    "é": "e",  # wtf didn't have this!
}

remove_diaeresis = {
    "ä": "a",
    "ë": "e",
    "ï": "i",
    "ö": "o",
    "ü": "u",
}
map_ligature = {
    "æ": "a",
    "œ": "o",
}

remove_reverse_hat = {
    "č": "c",
    "ŭ": "u",
}

strange_stuff = {
    # circle, bolle
    "å": "a",
    "ø": "o",
    "ç": "c",
    "ß": "s",  # TODO: one or two s?
}

all_kinds_of_apostrophes = "'’‘`´ʹʻʼʽʿˈ"  # also map itself, -> identity mapping to overwrite potentially removing mappings that came before
NORMALIZE_APOSTROPHES = {c: "'" for c in all_kinds_of_apostrophes}
NORMALIZE_DASH = {"–": "-", "-": "-"}  # utf8: b'\xe2\x80\x93'

REMOVE_EVERYTHING = (
    remove_backward_accent
    | remove_hats
    | remove_tilde
    | remove_flat
    | remove_accent
    | remove_diaeresis
    | map_ligature
    | remove_reverse_hat
    | strange_stuff
)

not_apostrophes_what_to_call_them = "„“”"
string_punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"  # string.punctuation
PUNCTUATION = string_punctuation + not_apostrophes_what_to_call_them
REPLACE_ALL_PUNCT_WITH_SPACE = {key: " " for key in PUNCTUATION}

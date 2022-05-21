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

ENGLISH_CHARACTER_MAPPING = (
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


PUNCTUATION = string.punctuation + "„“’”"
PUNCTUATION_MAPPING = {key: " " for key in PUNCTUATION}

PUNCTUATION_MAPPING.pop("-")
PUNCTUATION_MAPPING.pop("'")  # keep "'" cause its in wav2vec2-vocab
PUNCTUATION_MAPPING["–"] = "-"  # utf8: b'\xe2\x80\x93'

remove_them = ["`", "’"]
for punct in remove_them:
    PUNCTUATION_MAPPING[punct] = ""

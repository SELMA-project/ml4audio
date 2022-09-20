# tilo very pragmatically just wrote down some easily-mappable letters, no need to be perfect, just used as backup
RECOVER_CYRILLIC = {
    # "jo": "ё",
    "a": "а",
    "b": "б",
    "v": "в",
    "g": "г",
    "d": "д",
    "e": "е",
    # "sh": "ж",
    # "s": "з",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "f": "ф",
    "h": "х",
    "z": "ц",
}  #  "ъ", "ы", "ь", "э",

NO_JO = {
    "ё": "е",
    "ë": "е",
}

if __name__ == "__main__":

    for k, v in NO_JO.items():
        print(f"{k}: {k.encode('utf-8')} -> {v}: {v.encode('utf-8')}")

    for k, v in RECOVER_CYRILLIC.items():
        print(f"{k}: {k.encode('utf-8')} -> {v}: {v.encode('utf-8')}")

import os

from nemo.collections.nlp.models import PunctuationCapitalizationModel

if __name__ == "__main__":
    """
    Nur leere Drohungen oder ein realistisches Szenario. Wirtschaftsminister Robert
    Habeck hält inzwischen nichts mehr für ausgeschlossen. Was würde es bedeuten,
    wenn Wladimir Putin beschließt, Deutschland das Gas über diese Pipeline
    abzudrehen? Die wichtigsten Antworten im Überblick.
    """
    query = (
        "nur leere drohungen oder ein realistisches szenario "
        "wirtschaftsminister robert habeck hält inzwischen nichts mehr für "
        "ausgeschlossen was würde es bedeuten wenn wladimir putin "
        "beschließt deutschland das gas über diese pipeline abzudrehen "
        "die wichtigsten antworten im überblick"
    )

    nemo_model = f"{os.environ['BASE_PATH']}/data/cache/PROCESSED_DATA/NEMO_MODELS/NemoTrainedPunctuationCapitalizationModel-deu-1421ee5c9e895d0334f3d3c8a93d21eda0de2c61/nemo_exp_dir/Punctuation_and_Capitalization/2022-03-21_18-05-06/checkpoints/Punctuation_and_Capitalization.nemo"
    print(f"{nemo_model=}")
    DE_model = PunctuationCapitalizationModel.restore_from(nemo_model)
    result = DE_model.add_punctuation_capitalization([query])
    print(f"DE-Query  : {query}")
    print(f"DE-Result : {result}")

import requests

default_query = (
    "nur leere drohungen oder ein realistisches szenario "
    "wirtschaftsminister robert habeck hält inzwischen nichts mehr für "
    "ausgeschlossen was würde es bedeuten wenn wladimir putin "
    "beschließt deutschland das gas über diese pipeline abzudrehen "
    "die wichtigsten antworten im überblick"
)

if __name__ == "__main__":
    file = "/nm-raid/audio/work/thimmelsba/data/cache/PROCESSED_DATA/NEMO_MODELS/NemoTrainedPunctuationCapitalizationModel-deu-1421ee5c9e895d0334f3d3c8a93d21eda0de2c61/nemo_exp_dir/model.nemo"
    f = open(file, "rb")
    port = 8000
    files = {"file": (f.name, f, "multipart/form-data")}
    requests.post(url=f"http://127.0.0.1:{port}/upload_modelfile", files=files)

    r = requests.post(
        url=f"http://127.0.0.1:{port}/predict", json={"text": default_query}
    )
    print(r.text)

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from data_io.readwrite_files import write_lines

from ml4audio.speaker_tasks.speaker_embedding_utils import (
    read_sel_from_rttm,
    format_rttm_lines,
)
from ml4audio.speaker_tasks.speechbrain_der import speechbrain_DER

if __name__ == "__main__":

    for endpoint in ["predict", "predict_unsegmented"]:
        audio_file = "nemo_diarization/tests/resources/oLnl1D6owYA.opus"
        rttm_ref = "nemo_diarization/tests/resources/oLnl1D6owYA_ref.rttm"

        SR = 16_000
        start_end_speaker = read_sel_from_rttm(rttm_ref)

        f = open(audio_file, "rb")
        files = {
            "file": (f.name, f, "multipart/form-data"),
        }
        if endpoint == "predict":
            files["segments"] = (
                None,
                json.dumps([(s, e) for s, e, _ in start_end_speaker]),
                "application/json",
            )
        port = 8001
        r = requests.post(f"http://localhost:{port}/{endpoint}", files=files)
        response = r.json()
        print(f"{response}")
        s_e_labels = [
            (d["start"], d["end"], d["label"]) for d in response["labeled_segments"]
        ]

        file_id = Path(audio_file).stem

        with NamedTemporaryFile(suffix=".rttm") as tmp_file:
            rttm_pred_file = tmp_file.name
            write_lines(rttm_pred_file, format_rttm_lines(s_e_labels, file_id=file_id))
            miss_speaker, fa_speaker, sers, ders = speechbrain_DER(
                rttm_ref,
                rttm_pred_file,
                ignore_overlap=True,
                collar=0.25,
                individual_file_scores=True,
            )
            print(f"{(miss_speaker, fa_speaker, sers, ders)=}")

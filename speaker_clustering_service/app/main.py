import os
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, Form
from misc_utils.dataclass_utils import (
    encode_dataclass,
)

from ml4audio.audio_utils.audio_io import ffmpeg_torch_load
from ml4audio.audio_utils.audio_segmentation_utils import (
    expand_merge_segments,
    merge_short_segments,
)
from ml4audio.speaker_tasks.speaker_clusterer import SpeakerClusterer

DEBUG = os.environ.get("DEBUG", "False").lower() != "false"
if DEBUG:
    print("DEBUGGING MODE")


app = FastAPI(debug=DEBUG)

inferencer: Optional[SpeakerClusterer] = None


SR = 16_000


@app.post("/predict")
async def upload_and_process_audio_file(
    file: UploadFile, segments: list[tuple[float, float]] = Form()
):
    """
    TODO(tilo): cannot go with normal sync def method, cause:
    fastapi wants to run things in multiprocessing-processes -> therefore needs to pickle stuff
    some parts of nemo cannot be pickled: "_pickle.PicklingError: Can't pickle <class 'nemo.collections.common.parts.preprocessing.collections.SpeechLabelEntity'>"
    """
    global inferencer

    if not file:
        raise HTTPException(status_code=400, detail="Audio bytes expected")

    def save_file(filename, data):
        with open(filename, "wb") as f:
            f.write(data)

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_original:
        # data_bytes = file.file.read() # if in synchronous context otherwise just file
        data_bytes = await file.read()  # if in Asynchronous context
        save_file(tmp_original.name, data_bytes)

        raw_audio = ffmpeg_torch_load(tmp_original.name, sample_rate=SR).numpy()
    # audio = cut_away_noise(raw_audio)
    audio = raw_audio.astype(np.float)

    s_e_times = expand_merge_segments(segments, max_gap_dur=0.7, expand_by=0.1)
    s_e_times = merge_short_segments(s_e_times, min_dur=1.5)
    print(f"got {len(s_e_times)} segments")
    s_e_audio = [((s, e), audio[round(s * SR) : round(e * SR)]) for s, e in s_e_times]
    assert all((len(a) > 1000 for (s, e), a in s_e_audio))

    s_e_labels, _ = inferencer.predict(s_e_audio)

    return {
        "filename": file.filename,
        "labeled_segments": [
            {"start": s, "end": e, "label": l} for s, e, l in s_e_labels
        ],
    }


@app.get("/get_inferencer_dataclass")
def get_inferencer_dataclass() -> Dict[str, Any]:
    global inferencer
    if inferencer is not None:
        d = encode_dataclass(inferencer)
    else:
        d = {"response": "no model loaded yet!"}
    return d


@app.get("/model_config")
def get_model_config() -> Dict[str, Any]:
    global inferencer
    if inferencer is not None:
        d = encode_dataclass(
            inferencer,
            skip_keys=[
                "_id_",
                "_target_",
                "finetune_master",
                "cache_base",
                "cache_dir",
                "prefix",
                "use_hash_suffix",
                "lm_data",
            ],
        )
    else:
        d = {"response": "no model loaded yet!"}
    return d


@app.get("/inferencer_config")
def get_model_config() -> Dict[str, Any]:
    global inferencer
    if inferencer is not None:
        d = encode_dataclass(
            inferencer,
            skip_keys=[
                "_id_",
                "_target_",
                # "finetune_master",
                "cache_base",
                "cache_dir",
                "prefix",
                "use_hash_suffix",
                # "lm_data",
            ],
        )
    else:
        d = {"response": "no model loaded yet!"}
    return d


@app.on_event("startup")
def startup_event():
    global inferencer, vad
    inferencer = SpeakerClusterer(model_name="ecapa_tdnn", metric="cosine").build()


if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=2700,
        reload=True if DEBUG else False
        # log_level="debug"
    )

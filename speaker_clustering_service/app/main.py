import json
import os
from typing import Any, Optional, Dict

import uvicorn
from beartype.door import is_bearable
from fastapi import FastAPI, UploadFile, Form
from misc_utils.dataclass_utils import (
    encode_dataclass,
)

from ml4audio.audio_utils.audio_segmentation_utils import (
    expand_merge_segments,
    merge_short_segments,
)
from ml4audio.speaker_tasks.speaker_clusterer import SpeakerClusterer
from ml4audio.service_utils.fastapi_utils import (
    read_uploaded_audio_file,
    get_full_model_config,
)

DEBUG = os.environ.get("DEBUG", "False").lower() != "false"
if DEBUG:
    print("DEBUGGING MODE")


app = FastAPI(debug=DEBUG)

inferencer: Optional[SpeakerClusterer] = None


SR = 16_000


@app.post("/predict")
async def upload_and_process_audio_file(file: UploadFile, segments: str = Form()):
    """
    TODO(tilo): cannot go with normal sync def method, cause:
    fastapi wants to run things in multiprocessing-processes -> therefore needs to pickle stuff
    some parts of nemo cannot be pickled: "_pickle.PicklingError: Can't pickle <class 'nemo.collections.common.parts.preprocessing.collections.SpeechLabelEntity'>"
    """
    segments = json.loads(segments)
    is_bearable(segments, list[list[float]])
    segments: list[tuple[float, float]] = [(s, e) for s, e in segments]
    global inferencer

    audio = await read_uploaded_audio_file(file)

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


@app.get("/inferencer_config")
def get_model_config() -> Dict[str, Any]:
    global inferencer
    if inferencer is not None:
        d = get_full_model_config(inferencer)

    else:
        d = {"response": "no model loaded yet!"}
    return d


@app.on_event("startup")
def startup_event():
    global inferencer
    inferencer = SpeakerClusterer(model_name="ecapa_tdnn", metric="cosine").build()


if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=2700,
        reload=True if DEBUG else False
        # log_level="debug"
    )

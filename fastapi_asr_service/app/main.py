# pylint: skip-file
import logging
import os
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Dict

import numpy as np
import uvicorn
from beartype import beartype
from fastapi import FastAPI, UploadFile, HTTPException

from fastapi_asr_service.app.fastapi_asr_service_utils import load_asr_inferencer, \
    load_vad_inferencer
from misc_utils.beartypes import NumpyFloat1DArray
from misc_utils.dataclass_utils import (
    encode_dataclass,
)

from ml4audio.asr_inference.hf_asr_pipeline import (
    HfAsrPipelineFromLogitsInferencerDecoder,
)
from ml4audio.audio_utils.audio_io import ffmpeg_torch_load
from nemo_vad.nemo_offline_vad import NemoOfflineVAD

DEBUG = os.environ.get("DEBUG", "False").lower() != "false"
if DEBUG:
    print("DEBUGGING MODE")

logger = logging.getLogger("websockets")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger.addHandler(logging.StreamHandler())

app = FastAPI(debug=DEBUG)

asr_inferencer: Optional[HfAsrPipelineFromLogitsInferencerDecoder] = None
vad: Optional[NemoOfflineVAD] = None

# if DEBUG:
#     shutil.rmtree("debug_wavs", ignore_errors=True)
#     os.makedirs("debug_wavs")


def userfriendly_inferencer_dict(inferencer):
    return encode_dataclass(
        inferencer,
        skip_keys=[
            "_id_",
            "_target_",
            "finetune_master",
            "cache_base",
            "cache_dir",
            "lm_data",
        ],
    )


SR = 16_000


@beartype
def cut_away_noise(array: NumpyFloat1DArray) -> NumpyFloat1DArray:
    global vad
    start_ends, probas = vad.predict(array)
    if len(start_ends) == 0:
        # assuming that VAD fugedup so fallback to no-vad
        contat_array = array
    else:
        contat_array = np.concatenate(
            [array[round(s * SR) : round(e * SR)] for s, e in start_ends], axis=0
        )
    return contat_array


@app.post("/transcribe")
def upload_and_process_audio_file(file: UploadFile):
    global asr_inferencer

    if not file:
        raise HTTPException(status_code=400, detail="Audio bytes expected")

    def save_file(filename, data):
        with open(filename, "wb") as f:
            f.write(data)

    with NamedTemporaryFile(delete=True) as tmp_original:
        save_file(tmp_original.name, file.read())

        audio = ffmpeg_torch_load(tmp_original.name, sample_rate=SR)
    audio = cut_away_noise(audio.numpy())
    prediction = asr_inferencer.predict(audio)
    return {"filename": file.filename} | prediction


@app.get("/get_inferencer_dataclass")
def get_inferencer_dataclass() -> Dict[str, Any]:
    global asr_inferencer
    if asr_inferencer is not None:
        d = encode_dataclass(asr_inferencer)
    else:
        d = {"response": "no model loaded yet!"}
    return d


@app.get("/model_config")
def get_model_config() -> Dict[str, Any]:
    global asr_inferencer
    if asr_inferencer is not None:
        d = encode_dataclass(
            asr_inferencer,
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
    global asr_inferencer
    if asr_inferencer is not None:
        d = encode_dataclass(
            asr_inferencer,
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
    global asr_inferencer, vad
    asr_inferencer= load_asr_inferencer()
    vad=load_vad_inferencer()


if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=2700,
        reload=True if DEBUG else False
        # log_level="debug"
    )

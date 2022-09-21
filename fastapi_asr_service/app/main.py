# pylint: skip-file
import os
from typing import Any, Optional, Dict

import numpy as np
import uvicorn
from beartype import beartype
from fastapi import FastAPI, UploadFile

from fastapi_asr_service.app.fastapi_asr_service_utils import (
    load_asr_inferencer,
    load_vad_inferencer,
)
from misc_utils.beartypes import NumpyFloat1D
from misc_utils.dataclass_utils import (
    encode_dataclass,
)
from ml4audio.asr_inference.asr_chunk_infer_glue_pipeline import Aschinglupi
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript, letter_to_words
from ml4audio.service_utils.fastapi_utils import read_uploaded_audio_file, \
    get_full_model_config
from nemo_vad.nemo_offline_vad import NemoOfflineVAD

DEBUG = os.environ.get("DEBUG", "False").lower() != "false"
if DEBUG:
    print("DEBUGGING MODE")

# logger = logging.getLogger("websockets")
# logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
# logger.addHandler(logging.StreamHandler())

app = FastAPI(debug=DEBUG)

asr_inferencer: Optional[Aschinglupi] = None
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
def cut_away_noise(array: NumpyFloat1D) -> NumpyFloat1D:
    global vad
    start_ends, probas = vad.predict(array)
    if len(start_ends) == 0:
        # assuming that VAD fugedup so fallback to no-vad
        noise_free_array = array
    else:
        noise_free_array = np.concatenate(
            [array[round(s * SR) : round(e * SR)] for s, e in start_ends], axis=0
        )
    return noise_free_array


@app.post("/transcribe")
async def upload_and_process_audio_file(file: UploadFile):
    """
    TODO(tilo): cannot go with normal sync def method, cause:
    fastapi wants to run things in multiprocessing-processes -> therefore needs to pickle stuff
    some parts of nemo cannot be pickled: "_pickle.PicklingError: Can't pickle <class 'nemo.collections.common.parts.preprocessing.collections.SpeechLabelEntity'>"
    """
    global asr_inferencer

    audio = await read_uploaded_audio_file(file)

    audio = cut_away_noise(audio)
    at: AlignedTranscript = asr_inferencer.transcribe_audio_array(audio)
    at.remove_unnecessary_spaces()
    tokens = letter_to_words(at.letters)
    # TODO: rename chunks to tokens or whatever, rename timestamp to timespan ?

    hf_format = {
        "text": at.text,
        "chunks": [
            {
                "text": "".join([l.letter for l in letters]),
                "timestamp": (
                    at.abs_timestamp(letters[0]),
                    at.abs_timestamp(letters[-1]),
                ),
            }
            for letters in tokens
        ],
    }
    return {"filename": file.filename} | hf_format


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
        d = get_full_model_config(asr_inferencer)
    else:
        d = {"response": "no model loaded yet!"}
    return d


@app.on_event("startup")
def startup_event():
    global asr_inferencer, vad
    asr_inferencer = load_asr_inferencer()
    vad = load_vad_inferencer()


if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=2700,
        reload=True if DEBUG else False
        # log_level="debug"
    )

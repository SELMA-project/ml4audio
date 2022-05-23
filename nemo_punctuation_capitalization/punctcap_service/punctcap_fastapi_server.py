import logging
import logging
import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, UploadFile
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from pydantic import BaseModel

from misc_utils.dataclass_utils import (
    encode_dataclass,
)
from misc_utils.utils import just_try

DEBUG = os.environ.get("DEBUG", "False").lower() != "false"
if DEBUG:
    print("DEBUGGING MODE")

logger = logging.getLogger("websockets")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger.addHandler(logging.StreamHandler())

app = FastAPI(debug=DEBUG)
inferencer: Optional[PunctuationCapitalizationModel] = None


@app.get("/get_inferencer_dataclass")
def get_inferencer_dataclass() -> Dict[str, Any]:
    global inferencer
    if inferencer is not None:
        d = encode_dataclass(inferencer)
    else:
        d = {"response": "no model loaded yet!"}
    return d


@app.post("/upload_modelfile/")
async def upload_modelfile(file: UploadFile):
    global inferencer

    def save_file(filename, data):
        with open(filename, "wb") as f:
            f.write(data)

    nemo_model_file = "model.nemo"
    save_file(nemo_model_file, await file.read())

    just_try(
        lambda: load_nemo_model(nemo_model_file), default=None, reraise=True
    )  # TODO: here some 400er error if model-file does not pass the sanity_check
    return {"filename": file.filename}


default_query = (
    "nur leere drohungen oder ein realistisches szenario "
    "wirtschaftsminister robert habeck hält inzwischen nichts mehr für "
    "ausgeschlossen was würde es bedeuten wenn wladimir putin "
    "beschließt deutschland das gas über diese pipeline abzudrehen "
    "die wichtigsten antworten im überblick"
)


class PunctuationCapitalizationRequest(BaseModel):
    # TODO(tilo): do I need pydantic at all if just simple str is sent?
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": default_query,
            }
        }


@app.post("/predict")  # TODO: response_model=SomeResponsePydanticDataModel
async def predict(req: PunctuationCapitalizationRequest):
    global inferencer

    result = inferencer.add_punctuation_capitalization([req.text])

    return {"text": result}


def load_nemo_model(nemo_model="model.nemo"):
    """
    Nur leere Drohungen oder ein realistisches Szenario. Wirtschaftsminister Robert
    Habeck hält inzwischen nichts mehr für ausgeschlossen. Was würde es bedeuten,
    wenn Wladimir Putin beschließt, Deutschland das Gas über diese Pipeline
    abzudrehen? Die wichtigsten Antworten im Überblick.
    """
    global inferencer
    inferencer = PunctuationCapitalizationModel.restore_from(nemo_model)
    result = inferencer.add_punctuation_capitalization([default_query])
    # print(f"{inferencer.cfg=}")
    # print(f"DE-Result : {result}")


# @app.on_event("startup")
# async def startup_event():
#     file = "model.nemo"
#     load_nemo_model(file)


if __name__ == "__main__":
    """
        #TODO: why is that necessary?
    export PYTHONPATH=${PYTHONPATH}:/nm-raid/audio/work/thimmelsba/iais_code/NeMo
    """
    uvicorn.run(
        "punctcap_fastapi_server:app",
        host="127.0.0.1",
        port=2700,
        # log_level="debug"
    )

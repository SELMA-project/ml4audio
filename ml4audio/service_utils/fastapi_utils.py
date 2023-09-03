from abc import abstractmethod
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Union

import numpy as np
from beartype import beartype
from fastapi import UploadFile, HTTPException
from starlette.datastructures import UploadFile as starlette_UploadFile

from misc_utils.beartypes import NpFloatDim1, NumpyFloat32_1D, Dataclass
from misc_utils.buildable import Buildable
from misc_utils.dataclass_utils import encode_dataclass, decode_dataclass


_UploadFile = Union[UploadFile, starlette_UploadFile]


@dataclass
class DictPredictor:
    @abstractmethod
    def predict(self, data: dict) -> dict:
        raise NotImplementedError


@dataclass
class DataclassPredictor(Buildable):
    @abstractmethod
    def predict(self, data: Dataclass) -> Dataclass:
        raise NotImplementedError


@dataclass
class DataclassEncoderDecoderPredictorWrapper(Buildable, DictPredictor):
    predictor: DataclassPredictor

    def predict(self, data: dict) -> dict:
        return encode_dataclass(self.predictor.predict(decode_dataclass(data)))


@beartype
async def read_uploaded_audio_file(
    file: _UploadFile, SR: int = 16000
) -> NumpyFloat32_1D:
    # TODO: cannot typehint from fastapi import UploadFile cause it hands in UploadFile from starlette!
    from ml4audio.audio_utils.audio_io import ffmpeg_torch_load

    if not file:
        raise HTTPException(status_code=400, detail="Audio bytes expected")

    def save_file(filename, data):
        with open(filename, "wb") as f:
            f.write(data)

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_original:
        # data_bytes = file.file.read() # if in synchronous context otherwise just file
        data_bytes = await file.read()  # if in Asynchronous context
        save_file(tmp_original.name, data_bytes)

        raw_audio = ffmpeg_torch_load(tmp_original.name, target_sample_rate=SR).numpy()
    audio = raw_audio.astype(np.float32)
    return audio


def get_full_model_config(asr_inferencer):
    return encode_dataclass(
        asr_inferencer,
        skip_keys=[
            "_id_",
            "_target_",
            "cache_base",
            "cache_dir",
            "prefix",
            "use_hash_suffix",
        ],
    )

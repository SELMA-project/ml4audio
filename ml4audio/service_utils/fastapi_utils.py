from tempfile import NamedTemporaryFile

import numpy as np
from beartype import beartype
from fastapi import UploadFile, HTTPException
from misc_utils.beartypes import NumpyFloat1D
from misc_utils.dataclass_utils import encode_dataclass

from ml4audio.audio_utils.audio_io import ffmpeg_torch_load
from speaker_clustering_service.app.main import SR


@beartype
async def read_uploaded_audio_file(file:UploadFile)->NumpyFloat1D:
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
    audio = raw_audio.astype(np.float)
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

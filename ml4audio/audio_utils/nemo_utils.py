import numpy as np
from beartype import beartype

from misc_utils.beartypes import NpFloatDim1
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils import logging
from nemo_vad.nemo_offline_vad import NemoOfflineVAD


@beartype
def load_EncDecSpeakerLabelModel(pretrained_model: str) -> EncDecSpeakerLabelModel:
    """
    based on: https://github.com/NVIDIA/NeMo/blob/ddd87197e94ca23ae54e641dc7784e64c00a43d6/examples/speaker_tasks/recognition/speaker_reco_finetune.py#L63
    """
    if pretrained_model.endswith(".nemo"):
        logging.info(f"Using local speaker model from {pretrained_model}")
        model = EncDecSpeakerLabelModel.restore_from(restore_path=pretrained_model)
    elif pretrained_model.endswith(".ckpt"):
        logging.info(f"Using local speaker model from checkpoint {pretrained_model}")
        model = EncDecSpeakerLabelModel.load_from_checkpoint(
            checkpoint_path=pretrained_model
        )
    else:
        logging.info("Using pretrained speaker recognition model from NGC")
        model = EncDecSpeakerLabelModel.from_pretrained(model_name=pretrained_model)
    return model


@beartype
def nemo_offline_vad_to_cut_away_noise(
    vad: NemoOfflineVAD, array: NpFloatDim1, SR: int = 16_000
) -> NpFloatDim1:
    start_ends, probas = vad.predict(array)
    if len(start_ends) == 0:
        # assuming that VAD fugedup so fallback to no-vad
        noise_free_array = array
    else:
        noise_free_array = np.concatenate(
            [array[round(s * SR) : round(e * SR)] for s, e in start_ends], axis=0
        )
    return noise_free_array

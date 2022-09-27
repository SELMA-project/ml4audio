from beartype import beartype
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils import logging


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

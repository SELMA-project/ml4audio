# based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/speaker_tasks/recognition/speaker_reco_finetune.py
import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_language_classification.prepare_lang_clf_splits import create_subset_manifest

seed_everything(42)


@hydra_runner(config_path="conf", config_name="titanet-finetune.yaml")
def main(cfg):
    labels = ["en", "de", "es", "ru", "pt", "fr", "it"]
    cfg.model.train_ds.manifest_filepath = f"{os.environ['BASE_PATH']}/data/AUDIO_DATA/lang_clf_data_7lanuages/train_manifest.jsonl"
    cfg.model.validation_ds.manifest_filepath = f"{os.environ['BASE_PATH']}/data/AUDIO_DATA/lang_clf_data_7lanuages/validation_manifest.jsonl"
    # cfg.model.test_ds.manifest_filepath = f"{os.environ['BASE_PATH']}/data/AUDIO_DATA/lang_clf_data_7lanuages/test_manifest.jsonl"
    cfg.model.decoder.num_classes = len(labels)

    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    trainer = pl.Trainer(**cfg.trainer)
    exp_man_cfg = cfg.get("exp_manager", None)
    _ = exp_manager(trainer, exp_man_cfg)
    mdl = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    mdl.maybe_init_from_pretrained_checkpoint(cfg)
    mdl.cfg.train_ds.labels = labels  # TODO(tilo):WTF! had to manually stick labels in there, so that I have "vocabulary" at inference-time

    trainer.fit(mdl)

    # if (
    #     hasattr(cfg.model, "test_ds")
    #     and cfg.model.test_ds.manifest_filepath is not None
    # ):
    #     trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator)
    #     if mdl.prepare_test(trainer):
    #         trainer.test(mdl)


if __name__ == "__main__":
    main()

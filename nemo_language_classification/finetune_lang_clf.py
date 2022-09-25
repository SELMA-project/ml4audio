# based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/speaker_tasks/recognition/speaker_reco_finetune.py

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

seed_everything(42)


@hydra_runner(config_path="conf", config_name="titanet-finetune.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    _ = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    speaker_model.maybe_init_from_pretrained_checkpoint(cfg)
    trainer.fit(speaker_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator)
        if speaker_model.prepare_test(trainer):
            trainer.test(speaker_model)


if __name__ == '__main__':
    main()

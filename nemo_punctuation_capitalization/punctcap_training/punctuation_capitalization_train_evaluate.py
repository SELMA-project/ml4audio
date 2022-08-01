# based on: https://github.com/NVIDIA/NeMo/blob/3d0c29a317b89b20c93757010db80271eeea6816/examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py

import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.collections.nlp.models.token_classification.punctuation_capitalization_config import (
    PunctuationCapitalizationConfig,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="punctuation_capitalization_config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(42)
    cfg = OmegaConf.merge(OmegaConf.structured(PunctuationCapitalizationConfig()), cfg)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    if not cfg.do_training and not cfg.do_testing:
        raise ValueError(
            "At least one of config parameters `do_training` and `do_testing` has to `true`."
        )
    if cfg.do_training:
        if cfg.model.get("train_ds") is None:
            raise ValueError(
                "`model.train_ds` config section is required if `do_training` config item is `True`."
            )
    if cfg.do_testing:
        if cfg.model.get("test_ds") is None:
            raise ValueError(
                "`model.test_ds` config section is required if `do_testing` config item is `True`."
            )

    if not isinstance(cfg.pretrained_model,str) or cfg.pretrained_model in ["null","None"]:
        logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")
        model = PunctuationCapitalizationModel(cfg.model, trainer=trainer)
    else:
        if os.path.exists(cfg.pretrained_model):
            model = PunctuationCapitalizationModel.restore_from(cfg.pretrained_model)
        elif (
            cfg.pretrained_model
            in PunctuationCapitalizationModel.get_available_model_names()
        ):
            model = PunctuationCapitalizationModel.from_pretrained(cfg.pretrained_model)
        else:
            raise ValueError(
                f"Provide path to the pre-trained .nemo file or choose from "
                f"{PunctuationCapitalizationModel.list_available_models()}"
            )

        if cfg.do_training:
            model.update_config_after_restoring_from_checkpoint(
                class_labels=cfg.model.class_labels,
                common_dataset_parameters=cfg.model.common_dataset_parameters,
                train_ds=cfg.model.get("train_ds") if cfg.do_training else None,
                validation_ds=cfg.model.get(
                    "validation_ds") if cfg.do_training else None,
                test_ds=cfg.model.get("test_ds") if cfg.do_testing else None,
                optim=cfg.model.get("optim") if cfg.do_training else None,
            )
            model.set_trainer(trainer)

            model.setup_training_data()
            model.setup_validation_data()
            model.setup_optimization()
        else:
            model.setup_test_data(cfg.model.get("test_ds") )
    if cfg.do_training:
        trainer.fit(model)
    if cfg.do_testing:
        trainer.test(model)


if __name__ == "__main__":
    main()

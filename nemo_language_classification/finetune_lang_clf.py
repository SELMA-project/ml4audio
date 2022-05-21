# based on: https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_recognition/speaker_reco_finetune.py
import hydra

import pytorch_lightning as pl

from nemo.utils.exp_manager import exp_manager

from nemo_language_classification.nemo_lang_clf import load_EncDecSpeakerLabelModel
from nemo_language_classification.prepare_lang_clf_splits import create_subset_manifest


@hydra.main(config_path="conf", config_name="lang_clf_SpeakerNet.yaml")
def main(cfg):
    labels = cfg.subset_labels

    train_manifest_jsonl = "train_manifest.jsonl"
    create_subset_manifest(
        cfg.model.train_ds.manifest_filepath,
        labels,
        train_manifest_jsonl,
        num_per_label=10_000,
    )
    val_manifest_jsonl = "validation_manifest.jsonl"
    create_subset_manifest(
        cfg.model.validation_ds.manifest_filepath,
        labels,
        val_manifest_jsonl,
        num_per_label=100,
    )

    # data = data_io.read_jsonl(cfg.model.train_ds.manifest_filepath)
    # labels = list(sorted(set([d["label"] for d in data])))
    cfg.model.train_ds.labels = labels
    cfg.model.decoder.num_classes = len(labels)

    cfg.model.train_ds.manifest_filepath = train_manifest_jsonl
    cfg.model.validation_ds.manifest_filepath = val_manifest_jsonl
    cfg.model.test_ds.manifest_filepath = val_manifest_jsonl
    # see SpeechLabel class which does it like this sorted(set(map(lambda x: x.label, data)))

    mdl = load_EncDecSpeakerLabelModel(cfg.pretrained_model)
    mdl.setup_finetune_model(cfg.model)

    mdl.cfg.train_ds.labels = labels  # TODO(tilo):WTF! had to manually stick labels in there, so that I have "vocabulary" at inference-time
    # mdl.cfg.decoder.num_classes=len(labels)
    finetune_trainer = pl.Trainer(**cfg.trainer)
    exp_manager(finetune_trainer, cfg.get("exp_manager", None))
    mdl.set_trainer(finetune_trainer)
    mdl.setup_optimization(cfg.model.optim)

    if cfg.freeze_encoder:
        for param in mdl.encoder.parameters():
            param.requires_grad = False

    finetune_trainer.fit(mdl)

    # if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
    #     gpu = 1 if cfg.trainer.gpus != 0 else 0
    #     trainer = pl.Trainer(gpus=gpu)
    #     if mdl.prepare_test(trainer):
    #         trainer.test(mdl)


if __name__ == "__main__":
    main()

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch
import wandb

from data_io.readwrite_files import write_file
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import UNDEFINED, _UNDEFINED
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from nemo_punctuation_capitalization.punctcap_training.nemo_punctcap_traindata import (
    NepucaData,
    NepucaSplit,
)
from nemo_punctuation_capitalization.punctcap_training.punctuation_tatoeba_data import (
    TatoebaWikipediaData,
    TatoebaMonolingualData,
    TatoebaLanguages,
)


@dataclass
class NemoTrainedPunctuationCapitalizationModel(CachedData):
    data: Union[_UNDEFINED, NepucaData] = UNDEFINED
    base_model: str = "bert-base-multilingual-uncased"
    cache_base: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "NEMO_PUNCTCAP_MODELS")
    )

    @property
    def name(self):
        return f"{self.data.name}"

    def _build_cache(self):
        process = subprocess.Popen(
            "/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        cmd = self.build_train_command()
        write_file(self.prefix_cache_dir("train_command.sh"), cmd)
        out, err = process.communicate(cmd.encode("utf-8"))
        process.terminate()  # just me being paranoid!
        process.kill()
        nemo_model = list(Path(self.nemo_exp_dir).rglob("*.nemo"))[0]
        shutil.move(str(nemo_model), self.model_file)
        shutil.rmtree(nemo_model.parent)

    @property
    def model_file(self):
        return f"{self.nemo_exp_dir}/model.nemo"

    @property
    def nemo_exp_dir(self):
        return self.prefix_cache_dir("nemo_exp_dir")

    def build_train_command(self):
        cmd = f"""
DATA_DIR={self.data.data_dir}
BASE_PATH=${{PWD}}

# model configuration
# MODEL_CONFIG_YAML=$BASE_PATH/punctcap_training/conf/punctuation_capitalization_config.yaml
# export PYTHONPATH=${{PYTHONPATH}}:$BASE_PATH/iais_code/NeMo

python nemo_punctuation_capitalization/punctcap_training/punctuation_capitalization_train_evaluate.py \
    +do_testing=true \
    exp_manager.exp_dir={self.nemo_exp_dir} \
    model.language_model.pretrained_model_name={self.base_model} \
    model.train_ds.ds_item=$DATA_DIR \
    model.train_ds.tokens_in_batch=500 \
    model.train_ds.text_file=$DATA_DIR/text_train.txt \
    model.train_ds.labels_file=$DATA_DIR/labels_train.txt \
    model.validation_ds.ds_item=$DATA_DIR \
    model.validation_ds.text_file=$DATA_DIR/text_dev.txt \
    model.validation_ds.labels_file=$DATA_DIR/labels_dev.txt \
    model.test_ds.ds_item=$DATA_DIR \
    model.test_ds.text_file=$DATA_DIR/text_dev.txt \
    model.test_ds.labels_file=$DATA_DIR/labels_dev.txt | tee {self.prefix_cache_dir('train.log')}
        """
        return cmd


def train_punctcap(lang_code="deu"):
    wikipedia_data = TatoebaWikipediaData(
        raw_data=TatoebaMonolingualData(
            base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
            file_name=f"{lang_code}.tar",
        )
    ).build()
    limit = 110_000

    if wikipedia_data.num_lines < limit:
        print(
            f"{lang_code=} has too few wikipedia-data, only {wikipedia_data.num_lines} lines"
        )
        return

    dev_size = min(10_000, round(0.1 * wikipedia_data.num_lines))

    processed_data = NepucaData(
        train_dev_data=NepucaSplit(
            name=wikipedia_data.name,
            raw_lines=wikipedia_data,
            limit=limit,
            dev_size=dev_size,
        ),
        max_seq_len=30,
    )
    trained_model = NemoTrainedPunctuationCapitalizationModel(
        data=processed_data, clean_on_fail=False
    ).build()
    # ret=subprocess.run(cmd, capture_output=True, shell=True)
    # print(ret.stdout.decode())
    torch.cuda.empty_cache()
    # TODO: not inferring here alleviates gpu-out-of-mem issue? -> strangely it did!
    # inference_model = PunctuationCapitalizationModel.restore_from(
    #     trained_model.model_file
    # )
    # processed_data.build()
    # original_lines = read_lines(f"{processed_data.data_dir}/original_train.txt")
    # train_lines = read_lines(f"{processed_data.data_dir}/text_train.txt")
    # for original, query in itertools.islice(zip(original_lines, train_lines), 0, 3):
    #     prediction = inference_model.add_punctuation_capitalization([query])
    #     print(f"{query=}")
    #     print(f"{prediction=}")
    #     print(f"{original=}")


if __name__ == "__main__":
    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")
    BASE_PATHES["processed_data"] = PrefixSuffix("cache_root", "PROCESSED_DATA")

    wandb.init(project="asr-inference", name="punctcap_training")

    # nemo_nlp.modules.get_pretrained_lm_models_list()
    lang_codes = TatoebaLanguages().build()
    print(list(lang_codes))
    # lang_codes = ["por", "eng", "deu", "fra", "spa", "ita", "rus", "lit"]
    for lang in ["deu"]:  #
        train_punctcap(lang)

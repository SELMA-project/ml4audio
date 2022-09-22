import itertools
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from random import shuffle
from typing import Union, Optional, Any, Iterable, Iterator

import pandas
import torch
import wandb

from data_io.readwrite_files import write_file
from misc_utils.buildable import Buildable, BuildableList
from misc_utils.cached_data import CachedData
from misc_utils.dataclass_utils import UNDEFINED, _UNDEFINED
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from nemo_punctuation_capitalization.punctcap_training.lenta_data import LentaData
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
    pretrained_model: Optional[str] = None
    do_training: bool = True
    cache_base: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "NEMO_PUNCTCAP_MODELS")
    )

    @property
    def name(self):
        pm=Path(self.pretrained_model).stem if self.pretrained_model is not None else ""
        return f"{self.data.name}-{Path(self.base_model).stem}{pm}"

    def _build_cache(self):
        process = subprocess.Popen(
            "/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        cmd = self.build_train_command()
        write_file(self.prefix_cache_dir("train_command.sh"), cmd)
        out, err = process.communicate(cmd.encode("utf-8"))
        process.terminate()  # just me being paranoid!
        process.kill()
        if self.do_training:
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
    pretrained_model={self.pretrained_model} \
    +do_training={'true' if self.do_training else 'false'} \
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


random.seed(42)


@dataclass
class MixedCorpus(Buildable, Iterable[str]):
    name: str
    corpora: BuildableList
    _lines: list[str] = field(init=False, default_factory=lambda: [])
    limit: Optional[int] = None

    def _build_self(self) -> Any:
        limit_each = (
            int(self.limit / len(self.corpora)) if self.limit is not None else None
        )
        for corpus in self.corpora:
            self._lines += list(itertools.islice(corpus, 0, limit_each))
        shuffle(self._lines)
        print(f"{pandas.Series([len(t) for t in self._lines]).describe().to_dict()=}")

    def __iter__(self) -> Iterator[str]:
        yield from self._lines

    # def __len__(self):
    #     return len(self._lines)


def prep_wiki_text_corpus(lang_code):
    wikipedia_data = TatoebaWikipediaData(
        raw_data=TatoebaMonolingualData(
            base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
            file_name=f"{lang_code}.tar",
        )
    ).build()

    # if wikipedia_data.num_lines < limit:
    #     print(
    #         f"{lang_code=} has too few wikipedia-data, only {wikipedia_data.num_lines} lines"
    #     )
    #     return None
    # else:
    return wikipedia_data


def train_punctcap(
    text_corpus, # type: TextCorpus
    base_model="bert-base-multilingual-uncased",
    limit=110_000,
    dev_size=10_000,
    pretrained_model:Optional[str]=None,
    do_training=True,
):

    processed_data = NepucaData(
        train_dev_data=NepucaSplit(
            name=text_corpus.name,
            raw_lines=text_corpus,
            limit=limit,
            dev_size=dev_size,
        ),
        max_seq_len=30,
    )
    trained_model = NemoTrainedPunctuationCapitalizationModel(
        pretrained_model=pretrained_model,
        base_model=base_model,
        do_training=do_training,
        data=processed_data,
        clean_on_fail=False,
        overwrite_cache=True,
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
    """
        Punctuation Report:
    label       precision recall f1
    O (pad_label) 99.27 99.28 99.28
    ,             93.62 92.25 92.93
    .             92.83 94.70 93.76
    ?             63.60 23.29 34.09
    --------------------------------------------
    macro avg     87.33 77.38 80.01
    weighted avg  98.34 98.34 98.34

    Capitalization Report:
    label       precision recall f1
    O (pad_label) 99.35 99.31 99.33
    U             96.51 96.67 96.59
    --------------------------------------------
    macro avg     97.93 97.99 97.96
    weighted avg  98.88 98.88 98.88

    """
    base_path = os.environ["BASE_PATH"]
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root
    BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")
    BASE_PATHES["processed_data"] = PrefixSuffix("cache_root", "PROCESSED_DATA")

    wandb.init(project="nemo-punctcap", name="punctcap_training")

    # nemo_nlp.modules.get_pretrained_lm_models_list()
    lang_codes = TatoebaLanguages().build()
    print(list(lang_codes))
    # lang_codes = ["por", "eng", "deu", "fra", "spa", "ita", "rus", "lit"]
    lang = "rus"
    tugtekins_russian_model = f"{base_path}/data/PUNCTUATION/RU.nemo"
    train_punctcap(
        text_corpus=MixedCorpus(
            name=f"wiki_lenta_random",
            corpora=BuildableList(
                [
                    TatoebaWikipediaData(
                        raw_data=TatoebaMonolingualData(
                            base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
                            file_name=f"{lang}.tar",
                        )
                    ),
                    LentaData(),
                ]
            ),
        ),
        pretrained_model=tugtekins_russian_model,
        do_training=False,
        limit=11_000_000,
        dev_size=1000_000
    )
    # train_punctcap(
    #     text_corpus=MixedCorpus(
    #         name=f"wiki_lenta_random",
    #         corpora=BuildableList(
    #             [
    #                 TatoebaWikipediaData(
    #                     raw_data=TatoebaMonolingualData(
    #                         base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
    #                         file_name=f"{lang}.tar",
    #                     )
    #                 ),
    #                 LentaData(),
    #             ]
    #         ),
    #     ),
    #     do_training=True,
    #     limit=1_100_000,
    #     dev_size=100_000
    #
    # )
    # languages = ["por", "eng", "deu", "fra", "spa", "ita", "lit"]
    # for lang in ["rus"]:  #
    #     train_punctcap(
    #         text_corpus=MixedCorpus(
    #             name=f"wiki-{lang}",
    #             corpora=BuildableList(
    #                 [
    #                     TatoebaWikipediaData(
    #                         raw_data=TatoebaMonolingualData(
    #                             base_url="https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28",
    #                             file_name=f"{lang}.tar",
    #                         )
    #                     ),
    #                 ]
    #             ),
    #         ),
    #         do_training=True,
    #         limit=1_100_000,
    #         dev_size=100_000,
    #     )

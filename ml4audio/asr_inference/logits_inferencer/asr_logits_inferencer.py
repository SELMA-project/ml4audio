import importlib
import json
import os
import shutil
from abc import abstractmethod

# >> start of dataclass-patch
# TODO: WTF! when py-file run from __main__ the dataclass_patch needs explicit call to work!
import transformers
from beartype.door import is_bearable

from misc_utils.beartyped_dataclass_patch import (
    beartype_all_dataclasses_of_this_files_parent,
)

beartype_all_dataclasses_of_this_files_parent(__file__)
# << end of dataclass-patch

from misc_utils.utils import slugify_with_underscores
from ml4audio.audio_utils.torchaudio_utils import torchaudio_resample

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, ClassVar, Annotated, Optional, Any, Type, TypeVar

import numpy as np
import torch
from beartype import beartype
from beartype.vale import IsAttr, IsEqual
from numpy import floating, int16
from numpy.typing import NDArray
from transformers import (
    set_seed,
    AutoModel,
    AutoProcessor,
    Wav2Vec2ForCTC,
    PreTrainedModel,
)

from data_io.readwrite_files import read_file
from misc_utils.beartypes import (
    NumpyFloat2DArray,
    NumpyFloat1DArray,
    TorchTensor2D,
    NeList,
    NeStr,
)
from misc_utils.buildable import Buildable
from misc_utils.buildable_data import BuildableData, SlugStr
from misc_utils.cached_data import CachedData
from misc_utils.cached_data_specific import CachedList
from misc_utils.dataclass_utils import (
    _UNDEFINED,
    UNDEFINED,
    CLASS_REF_NO_INSTANTIATE,
    decode_dataclass,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from ml4audio.asr_inference.pytorch_to_onnx_for_wav2vec import (
    convert_to_onnx,
    quantize_onnx_model,
    WeightTypeName,
)
from ml4audio.audio_utils.audio_io import MAX_16_BIT_PCM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NumpyFloatORInt16_1DArray = Annotated[
    Union[NDArray[floating], NDArray[int16]], IsAttr["ndim", IsEqual[1]]
]

set_seed(42)


def _export_model_files_from_checkpoint_dir(model_name_or_path: str, model_dir: str):
    # TODO: does currently not work for pretrained models from hub
    os.makedirs(model_dir, exist_ok=True)
    needed_files = [
        "config.json",
        "pytorch_model.bin",
    ]
    for f in needed_files:
        file = f"{model_name_or_path}/{f}"
        assert os.path.isfile(file)
        shutil.copy(file, f"{model_dir}/")
    could_be_one_dir_above = [
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
    ]
    for f in could_be_one_dir_above:
        file = f"{model_name_or_path}/{f}"
        if not os.path.isfile(file):
            file = f"{model_name_or_path}/../{f}"
        if os.path.isfile(file):
            assert os.path.isfile(file), f"could not find {file}"
            shutil.copy(file, f"{model_dir}/")
        else:
            allowed_to_be_missing = ["special_tokens_map.json"]
            assert f in allowed_to_be_missing
    # TODO: no need for special permissions it is just a cache!
    # os.system(f"chmod -R 555 {model_dir}")


@dataclass
class HfCheckpoint(CachedData):
    """
    # TODO: I want to replace this by HfModelFromCheckpoint
    """

    # but need to think about __post_init__
    # and name parameter, which is more strict in HfModelFromCheckpoint
    name: Union[_UNDEFINED, NeStr] = UNDEFINED
    model_name_or_path: Optional[NeStr] = None

    cache_base: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["am_models"])

    @property
    def model_path(self) -> str:
        model_folder = self.model_dir
        if os.path.isdir(model_folder):
            return model_folder
        else:
            return self.model_name_or_path

    @property
    def model_dir(self):
        return self.prefix_cache_dir("model")

    def _build_cache(self):
        if os.path.isdir(self.model_name_or_path):
            _export_model_files_from_checkpoint_dir(
                self.model_name_or_path, self.model_dir
            )
        else:
            pass

    def __post_init__(self):
        if self.model_name_or_path is None:
            self.model_name_or_path = self.name
        self.name = self.name.replace("/", "_")
        assert len(self.name) > 0


HfPretrainedModelType = str


@dataclass
class HfModelFromCheckpoint(BuildableData):
    name: NeStr = UNDEFINED
    model_name_or_path: Optional[NeStr] = None
    # hf_model_type is necessary cause sometimes model-names are ambiguous, for "facebook/wav2vec2-base-960h" AutoModel resolves to the pretrained wav2vec2 without ctc-finetuning
    hf_model_type: HfPretrainedModelType = UNDEFINED
    base_dir: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["am_models"])

    def __post_init__(self):
        if self.model_name_or_path is None:
            self.model_name_or_path = self.name
        self.name = slugify_with_underscores(self.name)
        is_bearable(self.name, SlugStr)
        assert len(self.name) > 0

    @property
    def model_dir(self) -> str:
        return self.data_dir

    @property
    def model_path(self) -> str:
        model_folder = self.model_dir
        if os.path.isdir(model_folder):
            return model_folder
        else:
            return self.model_name_or_path

    @property
    def _is_data_valid(self) -> bool:
        """
        TODO: what about these?
            ├── config.json
            ├── preprocessor_config.json
            ├── pytorch_model.bin
            ├── special_tokens_map.json
            ├── tokenizer_config.json
            └── vocab.json
        """
        is_valid = all(
            (
                os.path.isfile(f"{self.data_dir}/{file}")
                for file in [
                    "pytorch_model.bin",
                    "config.json",
                    "preprocessor_config.json",
                ]
            )
        )
        return is_valid

    def _build_data(self) -> Any:

        if os.path.isdir(self.model_name_or_path):
            _export_model_files_from_checkpoint_dir(
                self.model_name_or_path, self.data_dir
            )
        else:
            clazz = getattr(importlib.import_module("transformers"), self.hf_model_type)
            clazz.from_pretrained(self.model_name_or_path).save_pretrained(
                self.data_dir
            )
            AutoProcessor.from_pretrained(self.model_name_or_path).save_pretrained(
                self.data_dir
            )


@dataclass
class FinetunedCheckpoint(HfCheckpoint):
    """
    kind of creepy that it is not supposed to be instantiated "normally" but only using this staticmethod (from_cache_dir)
    """

    finetune_master: Union[_UNDEFINED, dict] = UNDEFINED

    @staticmethod
    def from_cache_dir(finetune_master_cache_dir: str) -> list["FinetunedCheckpoint"]:
        s = read_file(f"{finetune_master_cache_dir}/dataclass.json")
        finetune_master: dict = json.loads(
            s.replace("_target_", CLASS_REF_NO_INSTANTIATE)
        )

        def get_finetune_task_cache_dir(fm):
            # TODO: this is very hacky!
            prefix_suffix = fm["finetune_client"]["task"]["cache_dir"]
            prefix_suffix["_target_"] = prefix_suffix.pop("_python_dataclass_")
            ps: PrefixSuffix = decode_dataclass(prefix_suffix)
            # dirr = f"{prefix_suffix['prefix']}/{prefix_suffix['suffix']}"
            return f"{ps}/output_dir"

        task_cache_dir = get_finetune_task_cache_dir(finetune_master)
        dcs = [str(p.parent) for p in Path(task_cache_dir).rglob("pytorch_model.bin")]

        if len(dcs) == 0:
            print(f"{task_cache_dir=} got no pytorch_model.bin -> remove it!!")

        def checkpoint_name_suffix(ckpt_dir):
            suff = ckpt_dir.replace(f"{task_cache_dir}/", "")
            return f"-{suff}" if ckpt_dir != task_cache_dir else ""

        return [
            FinetunedCheckpoint(
                name=f"{finetune_master['name']}{checkpoint_name_suffix(ckpt_dir)}",
                finetune_master=finetune_master,
                model_name_or_path=ckpt_dir,
            )
            for ckpt_dir in dcs
        ]


@dataclass
class OnnxedHFCheckpoint(HfCheckpoint):
    vanilla_chkpt: HfCheckpoint = UNDEFINED
    do_quantize: bool = True
    weight_type_name: Optional[WeightTypeName] = None
    name: NeStr = field(init=False)

    def __post_init__(self):
        suffix = "-quantized" if self.do_quantize else ""
        self.name = f"{self.vanilla_chkpt.name}-onnx{suffix}"
        super().__post_init__()

    @property
    def model_path(self) -> str:
        return self.vanilla_chkpt.model_path

    @property
    def onnx_model(self) -> str:
        return self._create_onnx_model_file(self.do_quantize)

    def _create_onnx_model_file(self, do_quantize: bool):
        suf = ".quant" if do_quantize else ""
        return self.prefix_cache_dir(f"wav2vec2{suf}.onnx")

    def _build_cache(self):
        model_id_or_path = self.vanilla_chkpt.model_path
        onnx_model_name = self._create_onnx_model_file(False)
        convert_to_onnx(model_id_or_path, onnx_model_name)

        if self.do_quantize:
            quantized_model_name = self._create_onnx_model_file(True)
            quantize_onnx_model(
                model_id_or_path=model_id_or_path,
                onnx_model_path=onnx_model_name,
                quantized_model_path=quantized_model_name,
                weight_type_name=self.weight_type_name,
            )
            os.remove(onnx_model_name)

        import onnx

        onnx_model = onnx.load(self.onnx_model)
        onnx.checker.check_model(self.onnx_model)


@dataclass
class ResamplingASRLogitsInferencer(Buildable):
    """
        Resampling  Asr Connectionis temporal classification (CTC) Logits Inference
        Reascoteclloin
        TODO: decouple resampling from asr-inferencer! who is dependency of whom?
        current issue: input_sample_rate triggers almost identical cache!

    ──────────────────────────────────────────────
    ──────│─────│───────│─────│───────│────────│──
    ──────│─────│───────│─────│───────│────────│──
    ──────│──┌───┬────┬───┐──┌┐───────│┌┐──────│──
    ──────│──│┌─┐│┌┐┌┐│┌─┐│──││───────┌┘└┐─────│──
    ──────│──││─└┴┘││└┤││└┘──││┌──┬──┬┼┐┌┼──┐──│──
    ──────│──││─┌┐─││─│││┌┬──┤││┌┐│┌┐├┤│││──┤──│──
    ──────│──│└─┘│─││─│└─┘├──┤└┤└┘│└┘│││└┼──│──│──
    ──────│──└───┘─└┘─└───┘──└─┴──┴─┐├┘└─┴──┘──│──
    ──────│─────│───────│─────│───┌─┘││────────│──
    ──────│─────│───────│─────│───└──┘│────────│──
    ──────│─────│───────│─────│───────│────────│──
    ──────│─────│───────│─────│───────│────────│──

    """

    checkpoint: Union[HfCheckpoint, HfModelFromCheckpoint] = UNDEFINED
    # transcript_normalizer: Union[_UNDEFINED, TranscriptNormalizer] = UNDEFINED
    input_sample_rate: int = 16000
    # https://stackoverflow.com/questions/61937520/proper-way-to-create-class-variable-in-data-class
    target_sample_rate: ClassVar[int] = 16000
    do_normalize: bool = True  # TODO: not sure yet what is better at inference time

    @property
    @beartype
    def name(self) -> NeStr:
        return self.checkpoint.name  # cut_away_path_prefixes(self.model_name)

    @property
    @abstractmethod
    def vocab(self) -> NeList[str]:
        raise NotImplementedError

    @beartype
    def resample_calc_logits(self, audio: NumpyFloatORInt16_1DArray) -> TorchTensor2D:
        logits = self.calc_logits(self._resample(audio))
        return logits

    @beartype
    def resample_calc_logsoftmaxed_logits(
        self, audio: NumpyFloatORInt16_1DArray
    ) -> NumpyFloat2DArray:
        return self.calc_logsoftmaxed_logits(self._resample(audio))

    @beartype
    def batch_resample_calc_logsoftmaxed_logits(
        self, audios: list[NumpyFloatORInt16_1DArray]
    ) -> tuple[list[NumpyFloat1DArray], list[NumpyFloat2DArray]]:
        resampled_audios = [self._resample(audio) for audio in audios]
        logits = self.batched_calc_logsoftmaxed_logits(resampled_audios)
        return resampled_audios, logits

    @beartype
    def _resample(self, audio: NumpyFloatORInt16_1DArray) -> NumpyFloat1DArray:
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / MAX_16_BIT_PCM
        if self.input_sample_rate != self.target_sample_rate:
            audio = torchaudio_resample(
                signal=torch.from_numpy(audio.astype(np.float32)),
                sample_rate=self.input_sample_rate,
                target_sample_rate=self.target_sample_rate,
            ).numpy()
        return audio

    @abstractmethod
    @beartype
    def calc_logits(self, audio: NumpyFloat1DArray) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @beartype
    def calc_logsoftmaxed_logits(self, audio: NumpyFloat1DArray) -> NumpyFloat2DArray:
        raise NotImplementedError

    @abstractmethod
    @beartype
    def batched_calc_logsoftmaxed_logits(
        self, audio: list[NumpyFloat1DArray]
    ) -> list[NumpyFloat2DArray]:
        raise NotImplementedError

    @staticmethod
    @beartype
    def calc_predicted_ids(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

    @staticmethod
    @beartype
    def logsoftmax(logits: TorchTensor2D) -> NumpyFloat2DArray:
        # CTC log posteriors inference
        with torch.no_grad():
            softmax = torch.nn.LogSoftmax(dim=-1)
            lpz = softmax(logits).cpu().numpy()
        return lpz


@dataclass
class VocabFromASRLogitsInferencer(CachedList):
    inferencer: Union[_UNDEFINED, ResamplingASRLogitsInferencer] = UNDEFINED

    cache_base: PrefixSuffix = field(
        default_factory=lambda: BASE_PATHES["processed_data"]
    )

    @property
    def name(self):
        return f"vocab-for-{self.inferencer.name}"

    @beartype
    def _build_data(self) -> NeList[str]:
        return self.inferencer.vocab


@dataclass
class VocabFromASRLogitsInferencerVolatile(Buildable, list[str]):
    inferencer: Union[_UNDEFINED, ResamplingASRLogitsInferencer] = UNDEFINED

    @beartype
    def _build_self(self) -> NeList[str]:
        assert len(self) == 0
        vocab = self.inferencer.vocab
        # if self.inferencer.name.startswith("jonatasgrosman"):
        #     # TODO: vocab fix, did huggingface change its logic? Orthography-thing?
        #     vocab = [c.upper() for c in vocab]
        # if "<s>" not in vocab:
        #     vocab.append("<s>")
        self.extend(vocab)
        return vocab


#
#
if __name__ == "__main__":

    # base_path = os.environ["BASE_PATH"]
    # cache_root = f"{base_path}/data/cache"
    # BASE_PATHES["cache_root"] = cache_root
    # BASE_PATHES["am_models"] = PrefixSuffix("cache_root", "AM_MODELS")
    model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")
    model.save_pretrained("./dings")
    model = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model.save_pretrained("./dings")

#
#     model=HfModelFromCheckpoint(
#         name="mpoyraz-wav2vec2-xls-r-300m-cv8-turkish",
#         model_name_or_path="mpoyraz/wav2vec2-xls-r-300m-cv8-turkish",
#     )
#     model=DockerImageAsVolumeWrapper(model)
#     print(f"{model.build()=}")

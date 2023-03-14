import json
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Union, Annotated, Optional, Sequence, ClassVar

import numpy as np
import soundfile
import torch
from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from omegaconf import DictConfig, OmegaConf

from misc_utils.beartypes import NumpyFloat1D, NeList, File
from misc_utils.buildable import Buildable
from misc_utils.buildable_data import BuildableData
from misc_utils.dataclass_utils import UNDEFINED
from misc_utils.prefix_suffix import PrefixSuffix
from misc_utils.utils import (
    set_val_in_nested_dict,
    get_val_from_nested_dict,
    NOT_EXISTING,
)
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_frame_pred,
    init_vad_model,
    prepare_manifest,
    load_tensor_from_file,
    prepare_gen_segment_table,
    generate_vad_segment_table_per_tensor,
)
from nemo.utils import logging

from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEnd,
    is_non_overlapping,
    expand_merge_segments,
)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"  # TODO!


import logging

logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)


@beartype
def create_manifest(
    manifest_file: Annotated[str, Is[lambda x: x.endswith("json")]],
    audio_file: File,
    rttm_file: Optional[File] = None,
) -> None:
    meta = {
        "audio_filepath": audio_file,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "uem_filepath": None,
    }
    if rttm_file:
        meta |= {"rttm_filepath": rttm_file}
    with open(manifest_file, "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")


VoiceSegments = Annotated[NeList[StartEnd], Is[is_non_overlapping]]

StartEndsVADProbas = tuple[VoiceSegments, list[float]]


@beartype
def nemo_offline_vad_infer(
    cfg: DictConfig, vad_model: EncDecClassificationModel, audio_file: File, tmpdir: str
) -> StartEndsVADProbas:
    """
    based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/asr/speech_classification/vad_infer.py
    TODO: get rid of all these stupid nemo file-read/writes! (manifest, table, ...)
    """

    data_dir = f"{tmpdir}"
    manifest_vad_input = f"{data_dir}/vad_manifest.json"
    create_manifest(manifest_vad_input, audio_file)  # TODO: use tmpfile!

    key_meta_map = {}
    with open(manifest_vad_input, "r") as manifest:
        for line in manifest.readlines():
            audio_filepath = json.loads(line.strip())["audio_filepath"]
            uniq_audio_name = audio_filepath.split("/")[-1].rsplit(".", 1)[0]
            if uniq_audio_name in key_meta_map:
                raise ValueError(
                    "Please make sure each line is with different audio_filepath! "
                )
            key_meta_map[uniq_audio_name] = {"audio_filepath": audio_filepath}

    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        config = {
            "input": manifest_vad_input,
            "window_length_in_sec": cfg.vad.parameters.window_length_in_sec,
            "split_duration": cfg.prepare_manifest.split_duration,
            "num_workers": cfg.num_workers,
            "prepared_manifest_vad_input": cfg.prepared_manifest_vad_input,
        }
        manifest_vad_input = prepare_manifest(config)
    else:
        logging.warning(
            "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
        )

    vad_model.setup_test_data(
        test_data_config={
            "vad_stream": True,  # TODO(tilo): whats the meaning of this?
            "sample_rate": 16000,
            "manifest_filepath": manifest_vad_input,
            "labels": [
                "infer",
            ],
            "num_workers": cfg.num_workers,
            "shuffle": False,
            "window_length_in_sec": cfg.vad.parameters.window_length_in_sec,
            "shift_length_in_sec": cfg.vad.parameters.shift_length_in_sec,
            "trim_silence": False,
            "normalize_audio": cfg.vad.parameters.normalize_audio,
        }
    )

    cfg.frame_out_dir = tmpdir
    start_end, probas = vad_inference_part(cfg, manifest_vad_input, vad_model)
    return start_end, probas


@beartype
def vad_inference_part(
    cfg: DictConfig, manifest_vad_input: File, vad_model: EncDecClassificationModel
) -> StartEndsVADProbas:
    if not os.path.exists(cfg.frame_out_dir):
        os.mkdir(cfg.frame_out_dir)
    else:
        logging.warning(
            "Note frame_out_dir exists. If new file has same name as file inside existing folder, it will append result to existing file and might cause mistakes for next steps."
        )
    logging.info("Generating frame level prediction ")
    vad_params = cfg.vad.parameters
    pred_dir = generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=vad_params.window_length_in_sec,
        shift_length_in_sec=vad_params.shift_length_in_sec,
        manifest_vad_input=manifest_vad_input,
        out_dir=cfg.frame_out_dir,
    )
    assert pred_dir == cfg.frame_out_dir
    logging.info(
        f"Finish generating VAD frame level prediction with window_length_in_sec={vad_params.window_length_in_sec} and shift_length_in_sec={vad_params.shift_length_in_sec}"
    )
    frame_length_in_sec = vad_params.shift_length_in_sec
    # overlap smoothing filter
    if vad_params.smoothing:
        # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
        # smoothing_method would be either in majority vote (median) or average (mean)
        logging.info("Generating predictions with overlapping input segments")
        smoothing_pred_dir = generate_overlap_vad_seq(
            frame_pred_dir=pred_dir,
            smoothing_method=vad_params.smoothing,
            overlap=vad_params.overlap,
            window_length_in_sec=vad_params.window_length_in_sec,
            shift_length_in_sec=vad_params.shift_length_in_sec,
            num_workers=cfg.num_workers,
            out_dir=cfg.smoothing_out_dir,
        )
        logging.info(
            f"Finish generating predictions with overlapping input segments with smoothing_method={vad_params.smoothing} and overlap={vad_params.overlap}"
        )
        pred_dir = smoothing_pred_dir
        frame_length_in_sec = 0.01
    suffixes = ("frame", "mean", "median")
    pred_filepath = [
        os.path.join(pred_dir, x) for x in os.listdir(pred_dir) if x.endswith(suffixes)
    ][0]
    per_args = {
        "frame_length_in_sec": frame_length_in_sec,
    }
    per_args |= vad_params.postprocessing
    sequence, name = load_tensor_from_file(pred_filepath)
    vad_probas = sequence.tolist()
    out_dir, per_args_float = prepare_gen_segment_table(sequence, per_args)
    preds = generate_vad_segment_table_per_tensor(sequence, per_args_float)
    preds = preds.detach().numpy().astype(np.float)
    return [(p[0], p[1]) for p in preds], vad_probas


DEFAULT_NEMO_VAD_CONFIG = {
    "name": "vad_inference_postprocessing",  # TODO: why this name?
    "dataset": None,
    "num_workers": 0,
    "sample_rate": 16000,
    "gen_seg_table": True,
    "write_to_manifest": True,
    "prepare_manifest": {"auto_split": True, "split_duration": 400},
    "vad": {
        "model_path": "vad_marblenet",
        "parameters": {
            "normalize_audio": False,
            "window_length_in_sec": 0.15,
            "shift_length_in_sec": 0.01,
            "smoothing": "median",
            "overlap": 0.875,
            "postprocessing": {
                "onset": 0.4,
                "offset": 0.7,  # TODO(tilo): makes no sense to me
                "pad_onset": 0.05,
                "pad_offset": -0.1,
                "min_duration_on": 0.2,
                "min_duration_off": 0.2,
                "filter_speech_first": True,
            },
        },
    },
    "prepared_manifest_vad_input": None,
    "frame_out_dir": "vad_frame",
    "smoothing_out_dir": None,
    "table_out_dir": None,
    "out_manifest_filepath": None,
}


@dataclass
class PathValue:
    path: list[str]
    value: Any


@dataclass
class NemoOfflineVAD(BuildableData):
    # for parameters see: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/asr/conf/vad/vad_inference_postprocessing.yaml
    name: str = UNDEFINED
    override_params: Optional[list[PathValue]] = None
    cfg: Union[dict, DictConfig] = field(
        default_factory=lambda: deepcopy(DEFAULT_NEMO_VAD_CONFIG)
    )
    min_gap_dur: float = 1.0
    expand_by: float = 0.5
    sample_rate: ClassVar[int] = 16000
    base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "MODELS/VAD_MODELS")
    )

    def __post_init__(self):
        if self.override_params is not None:
            for pv in self.override_params:
                val = get_val_from_nested_dict(self.cfg, pv.path)
                to_override_it_must_exist = val is not NOT_EXISTING
                assert to_override_it_must_exist, f"{pv.path=} does not exist"
                set_val_in_nested_dict(self.cfg, pv.path, pv.value)

        self.dictcfg = (
            OmegaConf.create(self.cfg) if isinstance(self.cfg, dict) else self.cfg
        )

    @property
    def _is_data_valid(self) -> bool:
        file = str(self.model_file)
        print(f"_is_data_valid: { file=}")
        return file.split(".")[-1] in ["nemo", "ckpt"] and is_bearable(file, File)

    def _build_data(self) -> Any:
        self._download_model()
        self._load_data()

    def _load_data(self):
        vad_model = init_vad_model(str(self.model_file))
        vad_model = vad_model.to(device)
        vad_model.eval()
        self.vad_model = vad_model

    @property
    def model_file_name(self) -> str:
        if self.dictcfg.vad.model_path.split(".")[-1] not in ["nemo", "ckpt"]:
            model_name = self.dictcfg.vad.model_path

        elif os.path.isfile(self.dictcfg.vad.model_path):
            source_file = self.dictcfg.vad.model_path
            model_name = source_file.split("/")[-1]
        else:
            raise NotImplementedError(f"{self.dictcfg.vad.model_path=}")
        return model_name

    @property
    def model_file(self) -> str:
        return f"{self.data_dir}/{self.model_file_name}"

    @beartype
    def _download_model(self) -> File:

        if self.dictcfg.vad.model_path.split(".")[-1] not in ["nemo", "ckpt"]:
            model_name = self.model_file_name
            (
                _,
                nemo_model_file_in_cache,
            ) = EncDecClassificationModel._get_ngc_pretrained_model_info(
                model_name=model_name
            )
            assert nemo_model_file_in_cache.endswith(model_name)
            source_file = nemo_model_file_in_cache

        elif os.path.isfile(self.dictcfg.vad.model_path):
            source_file = self.dictcfg.vad.model_path
        else:
            raise NotImplementedError(f"{self.dictcfg.vad.model_path=}")

        shutil.copy(source_file, self.model_file)
        return self.model_file

    @beartype
    def predict(self, audio: NumpyFloat1D) -> StartEndsVADProbas:
        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as tmpfile, tempfile.TemporaryDirectory(
            prefix="nemo_wants_to_write_many_files"
        ) as tmpdir:
            soundfile.write(tmpfile.name, audio, samplerate=self.sample_rate)
            # with open('unwanted_nemo.log', 'w') as sys.stdout: # not working!
            segments, probas = nemo_offline_vad_infer(
                self.dictcfg, self.vad_model, tmpfile.name, tmpdir
            )
        segments = expand_merge_segments(
            segments, min_gap_dur=self.min_gap_dur, expand_by=self.expand_by
        )
        return segments, probas


if __name__ == "__main__":
    model_name = "vad_marblenet"
    (
        class_,
        nemo_model_file_in_cache,
    ) = EncDecClassificationModel._get_ngc_pretrained_model_info(model_name=model_name)
    shutil.copy(nemo_model_file_in_cache, f"./")

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile
import torch
from beartype import beartype
from omegaconf import DictConfig

from misc_utils.beartypes import NumpyFloat1D, NeList
from misc_utils.buildable import Buildable
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

from ml4audio.audio_utils.audio_segmentation_utils import StartEnd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_manifest(manifest_file:str, audio_file):
    meta = {
        "audio_filepath": audio_file,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        # "rttm_filepath": rttm_file,
        "uem_filepath": None,
    }
    with open(manifest_file, "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")


StartEnds = NeList[StartEnd]
StartEndsVADProbas = tuple[StartEnds, list[float]]


@beartype
def nemo_offline_vad_infer(
    cfg: DictConfig, vad_model: EncDecClassificationModel, audio_file: str, tmpdir: str
) -> StartEndsVADProbas:
    """
    based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/asr/speech_classification/vad_infer.py
    TODO: get rid of all these stupid nemo file-read/writes! (manifest, table, ...)
    """
    assert os.path.isfile(audio_file)

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
def vad_inference_part(cfg, manifest_vad_input, vad_model) -> StartEndsVADProbas:
    if not os.path.exists(cfg.frame_out_dir):
        os.mkdir(cfg.frame_out_dir)
    else:
        logging.warning(
            "Note frame_out_dir exists. If new file has same name as file inside existing folder, it will append result to existing file and might cause mistakes for next steps."
        )
    logging.info("Generating frame level prediction ")
    pred_dir = generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
        shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
        manifest_vad_input=manifest_vad_input,
        out_dir=cfg.frame_out_dir,
    )
    assert pred_dir == cfg.frame_out_dir
    logging.info(
        f"Finish generating VAD frame level prediction with window_length_in_sec={cfg.vad.parameters.window_length_in_sec} and shift_length_in_sec={cfg.vad.parameters.shift_length_in_sec}"
    )
    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec
    # overlap smoothing filter
    if cfg.vad.parameters.smoothing:
        # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
        # smoothing_method would be either in majority vote (median) or average (mean)
        logging.info("Generating predictions with overlapping input segments")
        smoothing_pred_dir = generate_overlap_vad_seq(
            frame_pred_dir=pred_dir,
            smoothing_method=cfg.vad.parameters.smoothing,
            overlap=cfg.vad.parameters.overlap,
            window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
            num_workers=cfg.num_workers,
            out_dir=cfg.smoothing_out_dir,
        )
        logging.info(
            f"Finish generating predictions with overlapping input segments with smoothing_method={cfg.vad.parameters.smoothing} and overlap={cfg.vad.parameters.overlap}"
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
    per_args |= cfg.vad.parameters.postprocessing
    sequence, name = load_tensor_from_file(pred_filepath)
    vad_probas = sequence.tolist()
    out_dir, per_args_float = prepare_gen_segment_table(sequence, per_args)
    preds = generate_vad_segment_table_per_tensor(sequence, per_args_float)
    preds = preds.detach().numpy().astype(np.float)
    return [(p[0], p[1]) for p in preds], vad_probas


@dataclass
class NemoOfflineVAD(Buildable):
    cfg: DictConfig

    def _build_self(self) -> Any:
        # torch.set_grad_enabled(False)
        vad_model = init_vad_model(self.cfg.vad.model_path)
        vad_model = vad_model.to(device)
        vad_model.eval()
        self.vad_model = vad_model

    @beartype
    def predict(self, audio: NumpyFloat1D) -> StartEndsVADProbas:
        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as tmpfile, tempfile.TemporaryDirectory(
            prefix="nemo_wants_to_write_many_files"
        ) as tmpdir:
            soundfile.write(tmpfile.name, audio, samplerate=16000)
            pred = nemo_offline_vad_infer(
                self.cfg, self.vad_model, tmpfile.name, tmpdir
            )
        return pred

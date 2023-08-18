from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import soundfile
from beartype import beartype
from omegaconf import OmegaConf

from data_io.readwrite_files import write_lines
from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEndArraysNonOverlap,
    NonOverlSegs,
    StartEndLabels,
)
from ml4audio.speaker_tasks.diarization.speaker_diarization_inferencer import (
    SpeakerDiarizationInferencer,
)
from ml4audio.speaker_tasks.speaker_embedding_utils import (
    format_rttm_lines,
    read_sel_from_rttm,
)
from nemo.collections.asr.models import ClusteringDiarizer
from nemo_vad.nemo_offline_vad import NemoOfflineVAD, create_manifest


@dataclass
class NemoVadDiarizer(SpeakerDiarizationInferencer):
    nemo_vad: NemoOfflineVAD
    # cfg:DictConfig

    def _build_self(self) -> Any:
        self.cfg = OmegaConf.load(
            "ml4audio/speaker_tasks/diarization/offline_diarization.yaml"
        )

    @beartype
    def _run_vad(self, s_e_a: StartEndArraysNonOverlap) -> NonOverlSegs:
        assert len(s_e_a) == 1
        SR = self.nemo_vad.sample_rate
        s, e, a = s_e_a[0]
        return [
            (s + vad_s, s + vad_e) for vad_s, vad_e in self.nemo_vad.predict(audio=a)[0]
        ]

    @beartype
    def predict(self, s_e_a: StartEndArraysNonOverlap) -> StartEndLabels:
        segments = self._run_vad(s_e_a)
        _, _, array = s_e_a[0]

        # tmpdir = f"{os.getcwd()}/nemo_diar"
        # os.makedirs(tmpdir)
        with TemporaryDirectory(prefix="/tmp/nemo_tmp_dir") as tmpdir:
            fileid = "audio"
            audio_file = f"{tmpdir}/{fileid}.wav"
            manifest_file = f"{tmpdir}/manifest.json"
            rttm_file = f"{tmpdir}/vad.rttm"

            soundfile.write(audio_file, array, samplerate=16000)
            # manifest_file = f"speaker_tasks/tests/resources/input_manifest.json"
            write_lines(
                rttm_file,
                format_rttm_lines(
                    [(s, e, "NOSPEAKER") for s, e in segments], file_id=fileid
                ),
            )

            create_manifest(manifest_file, audio_file, rttm_file)
            self.cfg.diarizer.manifest_filepath = manifest_file
            self.cfg.diarizer.vad.model_path = None
            self.cfg.diarizer.speaker_embeddings.model_path = (
                "titanet-large"  # TODO: wtf hardcoded!
            )
            self.cfg.diarizer.oracle_vad = True
            # self.cfg.diarizer.vad.parameters.window_length_in_sec = params["window"]
            # self.cfg.diarizer.vad.parameters.shift_length_in_sec = params["step_dur"]
            self.cfg.diarizer.out_dir = tmpdir
            self.cfg.device = "cpu"
            sd_model = ClusteringDiarizer(cfg=self.cfg)
            sd_model.diarize()

            rttm_file = next(iter(Path(f"{tmpdir}/pred_rttms").glob("*.rttm")))
            return read_sel_from_rttm(str(rttm_file))

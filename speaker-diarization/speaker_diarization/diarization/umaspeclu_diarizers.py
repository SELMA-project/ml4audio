import os
from dataclasses import dataclass

from beartype import beartype

from misc_utils.prefix_suffix import BASE_PATHES
from ml4audio.asr_inference.inference import SetupTearDown
from ml4audio.audio_utils.audio_io import ffmpeg_load_trim
from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEndArraysNonOverlap,
    StartEndLabelNonOverlap,
)
from ml4audio.speaker_tasks.diarization.speaker_diarization_inferencer import (
    SpeakerDiarizationInferencer,
)
from ml4audio.speaker_tasks.nemo_speaker_embedder import NemoAudioEmbedder
from ml4audio.speaker_tasks.speaker_clusterer import UmascanSpeakerClusterer
from ml4audio.speaker_tasks.speaker_embedding_utils import read_sel_from_rttm
from nemo_vad.nemo_offline_vad import NemoOfflineVAD, PathValue


@dataclass
class UmaspecluDiarizer(SpeakerDiarizationInferencer):
    clusterer: UmascanSpeakerClusterer

    @beartype
    def predict(self, s_e_a: StartEndArraysNonOverlap) -> StartEndLabelNonOverlap:
        s_e_labels, _ = self.clusterer.predict(
            s_e_audio=[(s, e, a) for s, e, a in s_e_a]
        )
        return s_e_labels


@dataclass
class VadUmaspecluDiarizer(SpeakerDiarizationInferencer, SetupTearDown):
    nemo_vad: NemoOfflineVAD
    clusterer: UmascanSpeakerClusterer

    @property
    def _is_ready(self) -> bool:
        return self.nemo_vad._is_ready and self.clusterer._is_ready

    def __enter__(self):
        self.nemo_vad.__enter__()
        self.clusterer.__enter__()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.nemo_vad.__exit__()
        self.clusterer.__exit__()

    @beartype
    def predict(
        self, start_end_arrays: StartEndArraysNonOverlap
    ) -> StartEndLabelNonOverlap:
        SR = self.nemo_vad.sample_rate
        vad_se_a = [
            (
                [
                    (s + vad_s, s + vad_e)
                    for vad_s, vad_e in self.nemo_vad.predict(audio=a)[0]
                ],
                a,
            )
            for s, e, a in start_end_arrays
        ]

        vad_sea = [
            (s, e, a[round(s * SR) : round(e * SR)])
            for ses, a in vad_se_a
            for s, e in ses
        ]
        self.vad_segments = [(s, e) for ses, a in vad_se_a for s, e in ses]
        s_e_labels, _ = self.clusterer.predict(
            s_e_audio=[(s, e, a) for s, e, a in vad_sea]
        )
        return s_e_labels


if __name__ == "__main__":
    base_path = os.environ.get("BASE_PATH", "/tmp")
    cache_root = f"{base_path}/data/cache"
    BASE_PATHES["base_path"] = base_path
    BASE_PATHES["cache_root"] = cache_root

    file = "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"
    array = ffmpeg_load_trim(file)
    dw_news_sels = read_sel_from_rttm("tests/resources/oLnl1D6owYA_ref.rttm")
    vad_params_postpro = ["vad", "parameters", "postprocessing"]

    inferencer = VadUmaspecluDiarizer(
        clusterer=UmascanSpeakerClusterer(
            embedder=NemoAudioEmbedder(
                model_name="titanet_large",
            ),
            metric="euclidean",
            calibration_speaker_data=[
                (
                    "tests/resources/oLnl1D6owYA.opus",
                    [dw_news_sels[k] for k in (0, 15)] + [(75.0, 85.0, "DW-jingle")],
                )
            ],
        ),
        nemo_vad=NemoOfflineVAD(
            name="nemo_vad",
            override_params=[
                PathValue(
                    ["vad", "model_path"],
                    f"{BASE_PATHES['base_path']}/data/cache/MODELS/VAD_MODELS/nemovad/vad_multilingual_marblenet.nemo",
                ),
                PathValue(vad_params_postpro + ["onset"], 0.1),  # vbx-repo: 0.8
                PathValue(vad_params_postpro + ["offset"], 0.05),
                PathValue(vad_params_postpro + ["min_duration_on"], 0.1),
                PathValue(vad_params_postpro + ["min_duration_off"], 1.0),
                # PathValue(["num_workers"], 4), # not working!
            ],
            min_gap_dur=0.2,
            expand_by=0.1,
        ),
    )
    inferencer.build()
    with inferencer:
        print(f"{inferencer.predict([(0.0,len(array)/16000,array)])=}")

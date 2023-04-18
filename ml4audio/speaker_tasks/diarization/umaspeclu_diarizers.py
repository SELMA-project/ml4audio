from dataclasses import dataclass

from beartype import beartype

from ml4audio.audio_utils.audio_segmentation_utils import StartEndArraysNonOverlap, \
    StartEndLabelNonOverlap
from ml4audio.speaker_tasks.diarization.speaker_diarization_inferencer import \
    SpeakerDiarizationInferencer
from ml4audio.speaker_tasks.speaker_clusterer import UmascanSpeakerClusterer
from nemo_vad.nemo_offline_vad import NemoOfflineVAD


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
class VadUmaspecluDiarizer(UmaspecluDiarizer):
    nemo_vad: NemoOfflineVAD

    @beartype
    def predict(self, s_e_a: StartEndArraysNonOverlap) -> StartEndLabelNonOverlap:
        SR = self.nemo_vad.sample_rate
        vad_se_a = [
            (
                [
                    (s + vad_s, s + vad_e)
                    for vad_s, vad_e in self.nemo_vad.predict(audio=a)[0]
                ],
                a,
            )
            for s, e, a in s_e_a
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

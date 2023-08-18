import os
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any

import soundfile
from beartype import beartype

from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEndArraysNonOverlap,
    StartEndLabels,
)
from ml4audio.speaker_tasks.diarization.speaker_diarization_inferencer import (
    SpeakerDiarizationInferencer,
)
from ml4audio.speaker_tasks.speaker_embedding_utils import read_sel_from_rttm
from pyannote.audio import Pipeline


@dataclass
class PyannoteDiarizer(SpeakerDiarizationInferencer):
    pipeline: Pipeline = field(init=False, repr=False)

    def _build_self(self) -> Any:
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=os.environ["PYANNOTE_TOKEN"],
        )

    @beartype
    def predict(self, s_e_a: StartEndArraysNonOverlap) -> StartEndLabels:
        assert len(s_e_a) == 1
        _, _, array = s_e_a[0]

        with TemporaryDirectory(prefix="/tmp/nemo_tmp_dir") as tmpdir:
            fileid = "audio"
            audio_file = f"{tmpdir}/{fileid}.wav"
            rttm_file = f"{tmpdir}/vad.rttm"

            soundfile.write(audio_file, array, samplerate=16000)
            diarization = self.pipeline(audio_file)
            with open(rttm_file, "w") as f:
                diarization.write_rttm(f)

            return read_sel_from_rttm(str(rttm_file))

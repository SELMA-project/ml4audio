from dataclasses import dataclass, field
from typing import Any, Optional

import hdbscan
import umap
from beartype import beartype
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_contiguous_stamps,
    merge_stamps,
)
from pytorch_lightning import seed_everything

from misc_utils.beartypes import NumpyFloat1DArray, NumpyFloat2DArray
from misc_utils.buildable import Buildable
from nemo_diarization.speaker_embedding_utils import get_nemo_speaker_embeddings

seed_everything(42)


@dataclass
class SpeakerClusterer(Buildable):
    model_name: str
    window: float = 1.5
    step_dur: float = 0.75
    metric: str = "euclidean"  # cosine
    _speaker_model: EncDecSpeakerLabelModel = field(init=False, repr=False)
    _embeds: NumpyFloat2DArray = field(
        init=False, repr=False
    )  # cause one could want to investigate those after running predict
    _s_e_mapped_labels: list[tuple[float, float, str]] = field(
        init=False, repr=False
    )  # cause one could want to investigate those after running predict

    def _build_self(self) -> Any:
        self._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.model_name
        )

    @beartype
    def predict(
        self,
        s_e_audio: list[tuple[float, float, NumpyFloat1DArray]],
        ref_labels: Optional[list[str]] = None,
    ) -> tuple[list[tuple[float, float, str]], Optional[list[str]]]:

        self._s_e_mapped_labels, s_e_mapped_ref_labels = self._calc_raw_labels(
            s_e_audio, ref_labels
        )
        assert len(self._s_e_mapped_labels) == self._embeds.shape[0], (
            len(self._s_e_mapped_labels),
            self._embeds.shape,
        )
        if ref_labels is None:
            s_e_mapped_ref_labels = None

        lines = [" ".join([str(s), str(e), l]) for s, e, l in self._s_e_mapped_labels]
        a = get_contiguous_stamps(lines)
        lines = merge_stamps(a)
        s_e_labels = [l.split(" ") for l in lines]
        s_e_labels = [(float(s), float(e), l) for s, e, l in s_e_labels]
        return s_e_labels, s_e_mapped_ref_labels

    @beartype
    def _calc_raw_labels(
        self,
        s_e_audio: list[tuple[float, float, NumpyFloat1DArray]],
        ref_labels: Optional[list[str]],
    ):
        if ref_labels:
            assert len(ref_labels) == len(s_e_audio)
        else:
            ref_labels = ["dummy" for _ in range(len(s_e_audio))]
        self._embeds, start_ends, mapped_ref_labels = self._extract_embeddings(
            s_e_audio, ref_labels
        )
        umap_labels = self._umpa_cluster(self._embeds)
        s_e_mapped_labels = [
            (s, e, f"speaker_{l}") for (s, e), l in zip(start_ends, umap_labels)
        ]
        return s_e_mapped_labels, mapped_ref_labels

    @beartype
    def _umpa_cluster(self, embeds: NumpyFloat2DArray) -> list[int]:
        clusterable_embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=10,
            random_state=42,
            metric=self.metric,  # TODO: what about cosine?
        ).fit_transform(embeds)
        umap_labels = hdbscan.HDBSCAN(
            # min_samples=10,
            min_cluster_size=3,
        ).fit_predict(clusterable_embedding)
        umap_labels = [int(l) for l in umap_labels.tolist()]
        return umap_labels

    @beartype
    def _extract_embeddings(
        self,
        s_e_audio: list[tuple[float, float, NumpyFloat1DArray]],
        ref_labels: Optional[list[str]],
    ) -> tuple[NumpyFloat2DArray, list[tuple[float, float]], list[str]]:
        SR = 16_000
        labeled_segments = [(a, s, e, l) for (s, e, a), l in zip(s_e_audio, ref_labels)]
        embeds, s_e_mapped_labels = get_nemo_speaker_embeddings(
            labeled_segments,
            sample_rate=SR,
            speaker_model=self._speaker_model,
            window=self.window,
            shift=self.step_dur,
            expand_by=0.0,
            batch_size=1,
        )
        start_ends = [(s, e) for s, e, _ in s_e_mapped_labels]
        mapped_ref_labels = [l for s, e, l in s_e_mapped_labels]
        return embeds, start_ends, mapped_ref_labels

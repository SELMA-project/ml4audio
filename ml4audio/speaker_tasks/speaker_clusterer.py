from dataclasses import dataclass, field
from typing import Any, Optional

# https://github.com/scikit-learn-contrib/hdbscan/issues/457#issuecomment-1006344406
# strange error: numpy.core._exceptions._UFuncNoLoopError: ufunc 'correct_alternative_cosine' did not contain a loop with signature matching types  <class 'numpy.dtype[float32]'> -> None
# strange solution: https://github.com/lmcinnes/pynndescent/issues/163#issuecomment-1025082538
import numba
import numpy as np
import pynndescent

from ml4audio.audio_utils.nemo_utils import load_EncDecSpeakerLabelModel


@numba.njit(fastmath=True)
def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result


pynn_dist_fns_fda = pynndescent.distances.fast_distance_alternatives
pynn_dist_fns_fda["cosine"]["correction"] = correct_alternative_cosine
pynn_dist_fns_fda["dot"]["correction"] = correct_alternative_cosine
# </end_of_strange_solution>

import hdbscan
import umap
from beartype import beartype

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_contiguous_stamps,
    merge_stamps,
)
from pytorch_lightning import seed_everything

from misc_utils.beartypes import (
    NumpyFloat2DArray,
    NeNumpyFloat1DArray,
    NeList,
    NumpyFloat2D,
    NumpyFloat1D,
)
from misc_utils.buildable import Buildable
from ml4audio.audio_utils.audio_segmentation_utils import StartEnd, StartEndLabels
from ml4audio.speaker_tasks.speaker_embedding_utils import (
    get_nemo_speaker_embeddings,
)

seed_everything(42)
StartEndLabel = tuple[float, float, str]  # TODO: why not using TimeSpan here?


@dataclass
class UmascanSpeakerClusterer(Buildable):
    """
    UmascanSpeakerClusterer ->Umap+HDBSCAN NeMo Embeddings Speaker Clusterer
    nick-name: umaspeclu

    ░▄▄▄▄░
    ▀▀▄██►
    ▀▀███►
    ░▀███►░█►
    ▒▄████▀▀

    """

    model_name: str
    window: float = 1.5
    step_dur: float = 0.75
    metric: str = "euclidean"  # cosine
    _speaker_model: EncDecSpeakerLabelModel = field(init=False, repr=False)
    _embeds: NumpyFloat2DArray = field(
        init=False, repr=False
    )  # cause one could want to investigate those after running predict
    cluster_sels: StartEndLabels = field(
        init=False, repr=False
    )  # segments which are used for clustering, labeled given by clustering-algorithm, one could want to investigate those after running predict

    def _build_self(self) -> Any:
        self._speaker_model = load_EncDecSpeakerLabelModel(self.model_name)

    @beartype
    def predict(
        self,
        s_e_audio: list[tuple[StartEnd, NumpyFloat1D]],
        ref_labels: Optional[NeList[str]] = None,
    ) -> tuple[list[StartEndLabel], Optional[list[str]]]:

        self.cluster_sels, ref_sels_projected_to_cluster_sels = self._calc_raw_labels(
            s_e_audio, ref_labels
        )
        assert len(self.cluster_sels) == self._embeds.shape[0], (
            len(self.cluster_sels),
            self._embeds.shape,
        )
        if ref_labels is None:
            ref_sels_projected_to_cluster_sels = None

        lines = [" ".join([str(s), str(e), l]) for s, e, l in self.cluster_sels]
        a = get_contiguous_stamps(lines)
        lines = merge_stamps(a)
        s_e_labels_rw = [l.split(" ") for l in lines]
        s_e_labels = [(float(s), float(e), l) for s, e, l in s_e_labels_rw]
        return s_e_labels, ref_sels_projected_to_cluster_sels

    @beartype
    def _calc_raw_labels(
        self,
        s_e_audio: NeList[tuple[StartEnd, NumpyFloat1D]],
        ref_labels: Optional[list[str]] = None,
    ):
        if ref_labels is not None:
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
    def _umpa_cluster(self, embeds: NumpyFloat2D) -> list[int]:
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
        s_e_audio: NeList[tuple[StartEnd, NeNumpyFloat1DArray]],
        ref_labels: NeList[str],
    ) -> tuple[NumpyFloat2D, NeList[StartEnd], NeList[str]]:
        SR = 16_000
        labeled_segments = [
            (a, startend, l) for (startend, a), l in zip(s_e_audio, ref_labels)
        ]
        embeds, s_e_mapped_labels = get_nemo_speaker_embeddings(
            labeled_segments,
            sample_rate=SR,
            speaker_model=self._speaker_model,
            window=self.window,
            shift=self.step_dur,
            batch_size=1,  # TODO: whatabout higher batch-sizes?
        )
        start_ends = [(s, e) for s, e, _ in s_e_mapped_labels]
        mapped_ref_labels = [l for s, e, l in s_e_mapped_labels]
        return embeds, start_ends, mapped_ref_labels

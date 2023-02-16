from dataclasses import dataclass, field
from typing import Any, Optional, ClassVar

# https://github.com/scikit-learn-contrib/hdbscan/issues/457#issuecomment-1006344406
# strange error: numpy.core._exceptions._UFuncNoLoopError: ufunc 'correct_alternative_cosine' did not contain a loop with signature matching types  <class 'numpy.dtype[float32]'> -> None
# strange solution: https://github.com/lmcinnes/pynndescent/issues/163#issuecomment-1025082538
import numba
import numpy as np
import pynndescent
from beartype.door import is_bearable

from ml4audio.audio_utils.audio_data_models import Seconds
from ml4audio.audio_utils.nemo_utils import load_EncDecSpeakerLabelModel
from ml4audio.audio_utils.torchaudio_utils import load_resample_with_torch


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
from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEnd,
    StartEndLabels,
    NonOverlSegs,
    merge_segments_of_same_label,
    fix_segments_to_non_overlapping,
    StartEndArraysNonOverlap,
)
from ml4audio.speaker_tasks.speaker_embedding_utils import (
    get_nemo_speaker_embeddings,
)

seed_everything(42)
StartEndLabel = tuple[float, float, str]  # TODO: why not using TimeSpan here?

LabeledArrays = NeList[tuple[NumpyFloat1D, str]]


@beartype
def start_end_array(labeled_arrays: LabeledArrays) -> StartEndArraysNonOverlap:
    SR = 16000
    some_gap = 123.0
    one_week = float(7 * 24 * 60 * 60)  # longer than any audio-file

    def _g():
        offset = one_week
        for a, l in labeled_arrays:
            dur = len(a) / SR
            yield (offset, offset + dur, a)
            offset += dur + some_gap

    out = list(_g())
    assert len(out) == len(labeled_arrays)
    return out


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
    window: Seconds = 1.5
    step_dur: Seconds = 0.75
    metric: str = "euclidean"  # cosine
    same_speaker_min_gap_dur: Seconds = 0.1  # TODO: maybe this is not the clusterers responsibility, but some diarizers?
    calibration_speaker_data: Optional[list[tuple[str, StartEndLabels]]] = None
    _calib_labeled_arrays: Optional[LabeledArrays] = field(
        init=False, repr=False, default=None
    )
    CALIB_LABEL_PREFIX: ClassVar[
        str
    ] = "CALIBRATION_SPEAKER"  # for for calibration of clustering algorithm

    _speaker_model: EncDecSpeakerLabelModel = field(init=False, repr=False)
    _embeds: NumpyFloat2DArray = field(
        init=False, repr=False
    )  # cause one could want to investigate those after running predict
    cluster_sels: StartEndLabels = field(
        init=False, repr=False
    )  # segments which are used for clustering, labeled given by clustering-algorithm, one could want to investigate those after running predict

    def _build_self(self) -> Any:
        self._speaker_model = load_EncDecSpeakerLabelModel(self.model_name)
        if self.calibration_speaker_data is not None:
            self._calib_labeled_arrays = self._load_adj_data()

    @beartype
    def _load_adj_data(self) -> LabeledArrays:
        SR = 16000
        labeled_arrays = []
        for audio_file, s_e_ls in self.calibration_speaker_data:
            array = (
                load_resample_with_torch(audio_file, target_sample_rate=SR)
                .numpy()
                .astype(np.float32)
            )
            labeled_arrays.extend(
                [
                    (
                        array[round(s * SR) : round(e * SR)],
                        f"{self.CALIB_LABEL_PREFIX}-{l}",
                    )
                    for s, e, l in s_e_ls
                ]
            )
        return labeled_arrays

    @beartype
    def predict(
        self,
        s_e_audio: list[tuple[StartEnd, NumpyFloat1D]],
        ref_labels: Optional[
            NeList[str]
        ] = None,  # TODO: remove this, was only necessary for debugging?
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

        s_e_fixed = fix_segments_to_non_overlapping(
            [(s, e) for s, e, _ in self.cluster_sels]
        )  # TODO: use instead of get_contiguous_stamps
        s_e_labels = merge_segments_of_same_label(
            [(s, e, l) for (s, e), (_, _, l) in zip(s_e_fixed, self.cluster_sels)],
            min_gap_dur=self.same_speaker_min_gap_dur,
        )
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

        if self._calib_labeled_arrays is not None:
            s_e_audio += start_end_array(self._calib_labeled_arrays)
            ref_labels += [l for _, l in self._calib_labeled_arrays]

        self._embeds, start_ends, mapped_ref_labels = self._extract_embeddings(
            s_e_audio, ref_labels
        )
        real_data_indizes = [
            k
            for k, l in enumerate(mapped_ref_labels)
            if not l.startswith(self.CALIB_LABEL_PREFIX)
        ]

        umap_labels = self._umpa_cluster(self._embeds)
        s_e_mapped_labels = [
            (s, e, f"speaker_{l}") for (s, e), l in zip(start_ends, umap_labels)
        ]
        s_e_mapped_labels_real = [s_e_mapped_labels[k] for k in real_data_indizes]
        mapped_ref_labels_real = [mapped_ref_labels[k] for k in real_data_indizes]
        return s_e_mapped_labels_real, mapped_ref_labels_real

    @beartype
    def _umpa_cluster(self, embeds: NumpyFloat2D) -> list[int]:
        # for parameters see: https://umap-learn.readthedocs.io/en/latest/clustering.html
        clusterable_embedding = umap.UMAP(
            n_neighbors=30,  # _neighbors value – small values will focus more on very local structure and are more prone to producing fine grained cluster structure that may be more a result of patterns of noise in the data than actual clusters. In this case we’ll double it from the default 15 up to 30.
            min_dist=0.0,  # it is beneficial to set min_dist to a very low value. Since we actually want to pack points together densely (density is what we want after all) a low value will help, as well as making cleaner separations between clusters. In this case we will simply set min_dist to be 0.
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

from abc import abstractmethod
from dataclasses import dataclass
from random import shuffle
from typing import Union

import torch
from beartype import beartype
from matplotlib import pyplot as plt
from sklearn import preprocessing

from misc_utils.beartypes import (
    NumpyFloat2DArray,
    NeList,
    NumpyFloat1D,
    File,
)
from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEnd,
    StartEndLabels,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_subsegments,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


@beartype
def format_rttm_lines(
    start_end_speaker: StartEndLabels,
    file_id="who_cares",  # -> DER cares
) -> NeList[str]:
    """
    nvidia/nemo-code is too stupid I had to copy/past+refactor this
    """
    lines = []
    for start, end, speaker in start_end_speaker:
        duration = float(end) - float(start)
        assert duration > 0
        start = float(start)
        log = rttm_line(start, duration, file_id, speaker)
        lines.append(log)
    return lines


@beartype
def rttm_line(start, duration, file_id, speaker):
    return "SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>".format(
        file_id, start, duration, speaker
    )


@beartype
def read_sel_from_rttm(rttm_filename: File) -> StartEndLabels:
    start_end_speaker = []
    with open(rttm_filename, "r") as f:
        for line in f.readlines():
            rttm = line.strip().split()
            start, end, speaker = (
                float(rttm[3]),
                float(rttm[4]) + float(rttm[3]),
                rttm[7],
            )
            start_end_speaker.append((start, end, speaker))
    return start_end_speaker


FloatInt = Union[float, int]

OUTSIDE = "OUTSIDE"


@beartype
def apply_labels_to_segments(
    s_e_labels: StartEndLabels,  # they should be non-overlapping! otherwise things get overwritten!
    # but some (voxconverse) reference-data is overlapping!
    new_segments: NeList[StartEnd],  # might be overlapping
    min_overlap=0.6,  # proportion of overlap
) -> NeList[str]:
    """
    TODO(tilo): think about this method!
    """

    def calc_rel_overlap(s, e, sl, el) -> float:
        lens = e - s
        return (min(e, el) - max(s, sl)) / lens

    labels = []
    label_start = -1
    label_end = -1
    c = 0
    for s, e in new_segments:
        time_stamp = (s + e) / 2  # in the middle
        if c < len(s_e_labels):
            while s >= label_end and c < len(s_e_labels):
                label_start, label_end, lp = s_e_labels[c]
                c += 1
        is_completely_inside = s >= label_start and e <= label_end

        if is_completely_inside:
            labels.append(lp)
        elif calc_rel_overlap(s, e, label_start, label_end) >= min_overlap:
            labels.append(lp)
        else:
            labels.append(OUTSIDE)
            # print(f"OUTSIDE: {s=},{e=},{time_stamp=},{label_start=},{label_end=}")

    assert len(labels) == len(new_segments)
    return labels


@dataclass
class SubSegment:
    """
    sub-segment of a segments that starts at offset
    """

    offset: float
    start_end: StartEnd
    audio_array: NumpyFloat1D
    label: str

    @property
    def start(self):
        return self.start_end[0]

    @property
    def end(self):
        return self.start_end[1]


@beartype
def calc_subsegments_for_clustering(
    chunks: NeList[NumpyFloat1D],
    labeled_segments: StartEndLabels,
    sample_rate: int,
    shift: float,
    window: float,
) -> NeList[SubSegment]:
    """
    # "sub"-segmentation is based on: https://github.com/NVIDIA/NeMo/blob/4f06f3458b3d4d5e8ed3f5174d84e255a526321a/nemo/collections/asr/models/clustering_diarizer.py#L428
    """
    SR: int = sample_rate
    sub_segs = [
        SubSegment(start, sub_startend, slice_me_nice(sub_startend, chunk, SR), label)
        for chunk, (start, end, label) in zip(chunks, labeled_segments)
        for s, d in get_subsegments(
            offset=0.0, window=window, shift=shift, duration=len(chunk) / SR
        )
        for sub_startend in [(s, s + d)]
    ]
    return sub_segs


@dataclass
class SignalEmbedder:
    @abstractmethod
    def predict(self, arrays: NeList[NumpyFloat1D]) -> NeList[NumpyFloat1D]:
        raise NotImplementedError




@beartype
def slice_me_nice(startend: StartEnd, array: NumpyFloat1D, SR: int) -> NumpyFloat1D:
    start, end = startend
    ffrom = max(0, round(start * SR))
    tto = min(len(array) - 1, round(end * SR))
    sliced = array[ffrom:tto]
    assert len(sliced) > 0
    return sliced


@beartype
def save_umap_to_png(embedding: NumpyFloat2DArray, labels: list[str], png_file: str):
    assert len(labels) == embedding.shape[0]

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    num_speakers = len(set(labels))

    fig = plt.figure(figsize=(9, 9), dpi=90)
    ax = plt.subplot(111)
    # fmt: off
    markers = ["o", "v", "*", "d", "P" ]  # TODO: only via for-loop, see: https://stackoverflow.com/questions/62886268/plotting-different-clusters-markers-for-every-class-in-scatter-plot
    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # fmt: on

    color_markers = [(c, m) for c in colors for m in markers]
    shuffle(color_markers)
    for k, l in enumerate(le.classes_):
        idx = [i for i, sp in enumerate(labels) if sp == l]
        print(f"{l}: {len(idx)}")
        cluster_data = embedding[idx, :]
        c, m = color_markers[k % len(color_markers)]
        _sc = plt.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            # c=c,
            # cmap="Spectral",
            s=100,
            facecolors="none",
            edgecolors=c,
            alpha=0.5,
            label=f"{l} ({len(idx)})",
            marker=m,
        )
        # sc.set_facecolor("none")
    plt.gca().set_aspect("equal", "datalim")
    # plt.colorbar(boundaries=np.arange(num_speakers + 1) - 0.5).set_ticks(
    #     labels)
    speakers = list(le.classes_)
    assert len(speakers) == num_speakers
    # print(f"{len(speakers)=}")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # see: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    plt.legend(
        bbox_to_anchor=(1.0, 0.6),
        ncol=1,
        loc="center left",
    )
    plt.title("UMAP projection of the Speakers", fontsize=24)
    plt.savefig(png_file)

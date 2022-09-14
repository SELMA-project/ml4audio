from typing import Union

import numpy as np
import torch
from beartype import beartype
from matplotlib import pyplot as plt
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_subsegments,
    embedding_normalize,
)
from sklearn import preprocessing
from torch import autocast
from tqdm import tqdm

from misc_utils.beartypes import NumpyFloat1DArray, NumpyFloat2DArray
from misc_utils.processing_utils import iterable_to_batches

DEVICE = "cuda"
if not torch.cuda.is_available():
    DEVICE = "cpu"


try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def format_rttm_lines(
    start_end_speaker: list[tuple[float, float, str]], some_id="who_cares"
) -> list[str]:
    """
    nvidia/nemo-code is too so I had to copy/past+refactor this
    """
    lines = []
    for start, end, speaker in start_end_speaker:
        duration = float(end) - float(start)
        start = float(start)
        log = rttm_line(start, duration, some_id, speaker)
        lines.append(log)
    return lines


def rttm_line(start, duration, some_id, speaker):
    return "SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>".format(
        some_id, start, duration, speaker
    )


StartEndLabels = list[tuple[float, float, str]]


@beartype
def read_rttm(rttm_filename: str) -> StartEndLabels:
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


@beartype
def apply_labels_to_segments(
    s_e_labels: list[tuple[FloatInt, FloatInt, str]],
    new_segments: list[tuple[FloatInt, FloatInt]],
    min_overlap=0.6,  # proportion of overlap
) -> list[str]:
    def calc_rel_overlap(s, e, sl, el) -> float:
        lens = e - s
        return (min(e, el) - max(s, sl)) / lens

    labels = []
    label_start = -1
    label_end = -1
    for s, e in new_segments:
        time_stamp = (s + e) / 2  # in the middle
        if len(s_e_labels) > 0:
            while s >= label_end:
                label_start, label_end, lp = s_e_labels.pop(0)
        is_completely_inside = s >= label_start and e <= label_end

        if is_completely_inside:
            labels.append(lp)
        elif calc_rel_overlap(s, e, label_start, label_end) >= min_overlap:
            labels.append(lp)
        else:
            labels.append("OUTSIDE")
            print(f"OUTSIDE: {s=},{e=},{time_stamp=},{label_start=},{label_end=}")

    assert len(labels) == len(new_segments)
    return labels


@beartype
def get_nemo_speaker_embeddings(
    labeled_segments: list[tuple[NumpyFloat1DArray, float, float, str]],
    sample_rate: int,
    speaker_model,
    window=1.5,
    shift=0.75,
    expand_by=1.0,
    batch_size=1,
) -> tuple[NumpyFloat2DArray, StartEndLabels]:
    """
    based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/speaker_tasks/recognition/extract_speaker_embeddings.py

    based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/nemo/collections/asr/models/clustering_diarizer.py#L329
    """
    SR = sample_rate
    speaker_model = speaker_model.to(DEVICE)
    speaker_model.eval()

    all_embs = []

    ss_start_dur_segment_label = [
        (start, s, d, segment, label)
        for segment, start, end, label in labeled_segments
        for s, d in get_subsegments(
            offset=0.0, window=window, shift=shift, duration=len(segment) / SR
        )
    ]
    expand_half = expand_by / 2  # tilo: this was just my stupid idea to try things out
    overlapchunks_labels = [
        (
            segment[
                max(0, round((s - expand_half) * SR)) : min(
                    len(segment) - 1, round((s + d + expand_half) * SR)
                )
            ],
            label,
        )
        for ss, s, d, segment, label in ss_start_dur_segment_label
    ]
    for test_batch in tqdm(
        iterable_to_batches(overlapchunks_labels, batch_size=batch_size)
    ):
        audio_tensors = [torch.from_numpy(x).to(DEVICE) for x, _ in test_batch]
        audio_signal_len = torch.as_tensor([len(a) for a in audio_tensors]).to(DEVICE)
        no_need_for_padding_cause_all_have_same_len = (
            len(set([len(a) for a, _label in test_batch])) == 1
        )
        assert no_need_for_padding_cause_all_have_same_len, set(
            [len(a) for a, _label in test_batch]
        )
        audio_tensor = torch.concat([x.unsqueeze(0) for x in audio_tensors], dim=0)
        with autocast(), torch.no_grad():
            _, embs = speaker_model.forward(
                input_signal=audio_tensor, input_signal_length=audio_signal_len
            )
            emb_shape = embs.shape[-1]
            embs = embs.view(-1, emb_shape)
            all_embs.extend(embs.cpu().detach().numpy())
        del test_batch

    all_embs = np.asarray(all_embs)
    all_embs = embedding_normalize(all_embs)
    start_dur_label = [
        (float(ss + s), float(ss + s + d), l)
        for ss, s, d, _, l in ss_start_dur_segment_label
    ]
    assert len(all_embs) == len(start_dur_label)
    return all_embs, start_dur_label


@beartype
def save_umap_to_png(embedding: NumpyFloat2DArray, labels: list[str], png_file: str):
    assert len(labels) == embedding.shape[0]

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    num_speakers = len(set(labels))

    fig = plt.figure(figsize=(9, 9), dpi=90)
    ax = plt.subplot(111)
    # fmt: off
    markers = ["o", "v", "*", "x", "|" ]  # TODO: only via for-loop, see: https://stackoverflow.com/questions/62886268/plotting-different-clusters-markers-for-every-class-in-scatter-plot
    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # fmt: on

    color_markers = [(c, m) for m in markers for c in colors]
    for k, l in enumerate(le.classes_):
        idx = [i for i, sp in enumerate(labels) if sp == l]
        print(f"{l}: {len(idx)}")
        cluster_data = embedding[idx, :]
        c, m = color_markers[k % len(color_markers)]
        sc = plt.scatter(
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
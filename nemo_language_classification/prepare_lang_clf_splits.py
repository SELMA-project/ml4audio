import os
from collections import Counter
from itertools import groupby
from random import shuffle
from typing import List

from beartype import beartype

from data_io.readwrite_files import write_jsonl, read_jsonl

OTHER = "OTHER"


def fix_path(d):
    original_filepath = d["audio_filepath"]

    if "huggingface/datasets" in original_filepath:
        _, audio_filepath_tail = original_filepath.split("huggingface/datasets")
    else:
        _, audio_filepath_tail = original_filepath.split("huggingface_cache/datasets")
    d["audio_filepath"] = f"{os.environ['HF_DATASETS_CACHE']}/{audio_filepath_tail}"
    return d

@beartype
def create_subset_manifest(
    base_manifest: str,
    only_these_labels: list[str],
    subset_manifest_file: str,
    num_per_label: int = 100,
) -> None:
    """
    base_manifest contains all data with many labels
    eventuall creates OTHER-class by randomly selecting from class-labels not contained in labels
    """
    data = list(read_jsonl(base_manifest))
    gr = (
        (k, list(g))
        for k, g in groupby(
            sorted(data, key=lambda x: x["label"]), lambda x: x["label"]
        )
    )
    lang2data = {k: g[:num_per_label] for k, g in gr}
    data = [d for k in only_these_labels if k != OTHER for d in lang2data.pop(k)]

    if OTHER in only_these_labels:
        other_data = [d for g in lang2data.values() for d in g]
        shuffle(other_data)
        for d in other_data:
            d["label"] = OTHER
        data += other_data[: len(only_these_labels) * num_per_label]

    print(subset_manifest_file)
    print(Counter(d["label"] for d in data))
    shuffle(data)
    write_jsonl(subset_manifest_file, map(fix_path, data))

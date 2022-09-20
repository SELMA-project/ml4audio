import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from data_io.readwrite_files import write_lines, read_json
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript, LetterIdx
from ml4audio.audio_utils.audio_io import (
    load_resample_with_nemo,
)
from nemo_diarization.audio_segmentation_utils import expand_segments, segment_by_pauses
from nemo_diarization.speaker_clusterer import SpeakerClusterer
from nemo_diarization.speaker_embedding_utils import (
    format_rttm_lines,
    read_sel_from_rttm,
    apply_labels_to_segments,
)
from nemo_diarization.speechbrain_der import speechbrain_DER


# @pytest.mark.skip
def test_speaker_clusterer_oracle_vad(
    rttm_ref="nemo_diarization/tests/resources/oLnl1D6owYA_ref.rttm",
    audio_file="nemo_diarization/tests/resources/oLnl1D6owYA.opus",
):

    SR = 16_000
    start_end_speaker = read_sel_from_rttm(rttm_ref)
    array = load_resample_with_nemo(audio_file)

    s_e_audio = [
        (s, e, array[round(s * SR) : round(e * SR)])
        for s, e, speaker in start_end_speaker
    ]

    clusterer: SpeakerClusterer = SpeakerClusterer(model_name="ecapa_tdnn").build()
    s_e_labels, _ = clusterer.predict(s_e_audio)
    with NamedTemporaryFile(suffix=".rttm") as tmp_file:
        rttm_pred_file = tmp_file.name
        write_lines(
            rttm_pred_file, format_rttm_lines(s_e_labels, file_id=Path(audio_file).stem)
        )
        miss_speaker, fa_speaker, sers, ders = speechbrain_DER(
            rttm_ref,
            rttm_pred_file,
            ignore_overlap=True,
            collar=0.25,
            individual_file_scores=True,
        )
        print(f"{(miss_speaker, fa_speaker, sers, ders)=}")

    speaker_confusion = float(sers[0])
    assert speaker_confusion < 2.1, speaker_confusion


TEST_RESOURCES = "tests/resources"
BASE_PATHES["test_resources"] = TEST_RESOURCES


def get_test_cache_base():
    cache_base = PrefixSuffix("test_resources", "cache")
    if os.path.isdir(str(cache_base)):
        shutil.rmtree(str(cache_base))
    os.makedirs(str(cache_base))
    return cache_base


cache_root = get_test_cache_base()
BASE_PATHES["asr_inference"] = cache_root
BASE_PATHES["base_path"] = cache_root
BASE_PATHES["cache_root"] = cache_root
# BASE_PATHES["raw_data"] = PrefixSuffix("cache_root", "RAW_DATA")
# BASE_PATHES["processed_data"] = PrefixSuffix("cache_root", "PROCESSED_DATA")
BASE_PATHES["am_models"] = PrefixSuffix("cache_root", "AM_MODELS")
BASE_PATHES["asr_inference"] = PrefixSuffix("cache_root", "ASR_INFERENCE")

# @pytest.mark.skip
def test_speaker_clusterer(
    rttm_ref="nemo_diarization/tests/resources/oLnl1D6owYA_ref.rttm",
    audio_file="nemo_diarization/tests/resources/oLnl1D6owYA.opus",
    aligned_transcript_file="nemo_diarization/tests/resources/aligned_transcript.json",
):
    # at = asr_infer(
    #     audio_file, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-english"
    # )
    # write_json(
    #     "nemo_diarization/tests/resources/aligned_transcript.json",
    #     to_dict(at),
    # )
    SR = 16000

    d = read_json(aligned_transcript_file)
    at = AlignedTranscript(
        letters=[LetterIdx(x["letter"], x["r_idx"]) for x in d.pop("letters")], **d
    )
    at.remove_unnecessary_spaces()
    # print(f"{at.text=}")
    s_e = segment_by_pauses(at, min_pause_dur=0.7)
    timestamps = at.abs_timestamps
    s_e_times = expand_segments(s_e, timestamps)
    for (s, e), (st, et) in zip(s_e, s_e_times):
        print(
            f"{timestamps[s]}->{timestamps[e]}\t{st}->{et}\t{et - st}\t{at.text[s:(e + 1)]}"
        )
    s_e_sp_ref = read_sel_from_rttm(rttm_ref)
    array = load_resample_with_nemo(audio_file)
    s_e_audio = [(s, e, array[round(s * SR) : round(e * SR)]) for s, e in s_e_times]
    assert all((len(a) > 1000 for s, e, a in s_e_audio))
    clusterer: SpeakerClusterer = SpeakerClusterer(model_name="ecapa_tdnn").build()
    s_e_labels, _ = clusterer.predict(s_e_audio)
    s_e_mapped_labels = clusterer._s_e_mapped_labels
    labels_ref = apply_labels_to_segments(
        s_e_sp_ref, [(s, e) for s, e, _ in s_e_mapped_labels]
    )
    labels_pred = [l for _, _, l in s_e_mapped_labels]
    rand_score, mutual_info_score = (
        adjusted_rand_score(labels_ref, labels_pred),
        adjusted_mutual_info_score(labels_ref, labels_pred),
    )
    expected_rand_score = 0.9880002978471785
    expected_mutinfo_score = 0.9697478261936556
    print(f"{rand_score=}~={expected_rand_score=}")
    print(f"{mutual_info_score=}~={expected_mutinfo_score=}")

    assert rand_score >= expected_rand_score
    assert mutual_info_score >= expected_mutinfo_score

    with NamedTemporaryFile(
        suffix=".rttm",
        dir="./",
        delete=True,
    ) as tmp_file:
        rttm_pred_file = tmp_file.name
        write_lines(
            rttm_pred_file, format_rttm_lines(s_e_labels, file_id=Path(audio_file).stem)
        )
        miss_speaker, fa_speaker, sers, ders = speechbrain_DER(
            rttm_ref,
            rttm_pred_file,
            ignore_overlap=True,
            collar=0.25,
            individual_file_scores=True,
        )
    print(f"{(miss_speaker, fa_speaker, sers, ders)=}")
    speaker_confusion = float(sers[0])
    diarization_error_rate = float(ders[0])
    assert speaker_confusion < 2.0, speaker_confusion
    assert diarization_error_rate < 18.0, diarization_error_rate


"""
umap_hdbscan: rand_score=0.9818405682347752,mutual_info_score=0.9624121516246783
(miss_speaker, fa_speaker, sers, ders)=(array([3.21688171, 3.21688171]), array([0.05671672, 0.05671672]), array([0.15002892, 0.15002892]), array([3.42362735, 3.42362735]))

"""
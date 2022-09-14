import os
import pickle
import sys
from beartype import beartype

from ml4audio.asr_inference.hfwav2vec2_asr_decode_inferencer import \
    HFASRDecodeInferencer
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWAV2VEC2_SAMPLE_RATE,
)
from ml4audio.asr_inference.transcript_glueing import glue_left_right
from ml4audio.audio_utils.aligned_transcript import (
    LetterIdx,
    NeAlignedTranscript,
)

import difflib
from typing import Optional, Iterable

import numpy as np
from nemo.collections.asr.parts.preprocessing import AudioSegment


@beartype
def glue_transcripts(
    aligned_transcripts: Iterable[NeAlignedTranscript],
    debug=False,
) -> NeAlignedTranscript:
    """
    TODO: when/where is this used?
    """

    sm = difflib.SequenceMatcher()
    sample_rate = None
    previous: Optional[NeAlignedTranscript] = None
    letters: list[LetterIdx] = []
    very_start = None
    for ts in aligned_transcripts:
        if sample_rate is None:
            sample_rate = ts.sample_rate
        if previous is not None:
            _, glued = glue_left_right(left=previous, right=ts, sm=sm)
            print(f"prev-offset: {glued.abs_idx(glued.letters[0])}")
            letters = [l for l in letters if l.r_idx < glued.abs_idx(glued.letters[0])]
            letters.extend(
                [LetterIdx(l.letter, glued.abs_idx(l)) for l in glued.letters]
            )
        else:
            if debug:
                print(f"initial: {ts.text}")
            very_start = ts.offset
            letters.extend([LetterIdx(x.letter, ts.abs_idx(x)) for x in ts.letters])
        previous = ts

    monotonously_increasing = all(
        (letters[k].r_idx >= letters[k - 1].r_idx for k in range(1, len(letters)))
    )
    assert (
        monotonously_increasing
    ), f"text: {''.join([l.letter for l in letters])},indizes: {[l.r_idx for l in letters]}"

    transcript = NeAlignedTranscript(letters, sample_rate, offset=very_start)
    return transcript


def generate_arrays(samples: np.ndarray, step):
    for idx in range(0, len(samples), step):
        segm_end_idx = round(idx + 2 * step)
        next_segment_too_small = len(samples) - segm_end_idx < step
        if next_segment_too_small:
            array = samples[idx:]  # merge this one with next
            yield idx, array
            break
        else:
            array = samples[idx:segm_end_idx]
            yield idx, array


@beartype
def transcribe_audio_file(
    asr: HFASRDecodeInferencer, file, step_dur=5, do_cache=False
) -> NeAlignedTranscript:
    """
    TODO: what is this good for?
    """
    audio = AudioSegment.from_file(
        file,
        target_sr=HFWAV2VEC2_SAMPLE_RATE,
        offset=0.0,
        trim=False,
    )
    step = round(HFWAV2VEC2_SAMPLE_RATE * step_dur)
    arrays = generate_arrays(audio.samples, step)

    cache_file = "aligned_transcripts.pkl"
    if not do_cache or not os.path.isfile(cache_file):

        aligned_transcripts = [
            asr.transcribe_audio_array(array) for idx, array in arrays
        ]
        with open(cache_file, "wb") as f:
            pickle.dump(aligned_transcripts, f)
    else:
        print(f"found cached {cache_file}")

    if do_cache:
        with open(cache_file, "rb") as f:
            aligned_transcripts = pickle.load(f)

    transcript = glue_transcripts(aligned_transcripts)
    return transcript

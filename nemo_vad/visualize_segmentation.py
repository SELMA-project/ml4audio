#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from audio_data.speech_utils import torchaudio_info

from nemo_vad.nemo_streaming_vad import NeMoVAD
from audio_utils.audio_streaming import (
    break_into_chunks,
    load_and_resample,
)


def offline_inference(vad: NeMoVAD, signal_chunks):
    preds = []
    proba_b = []
    proba_s = []

    for signal in signal_chunks:
        result = vad.predict(signal)

        preds.append(result.label_id)
        proba_b.append(result.probs_background)
        proba_s.append(result.probs_speech)

    vad.reset()

    return preds, proba_b, proba_s


def visualize(results, audio, sample_rate, threshold, dur):
    import librosa.display

    plt.figure(figsize=[20, 10])
    num = len(results)
    for i, (FRAME_LEN, buffer_size, _, _, proba_s) in enumerate(results):
        len_pred = len(results[i][2])
        ax1 = plt.subplot(num + 1, 1, i + 1)

        ax1.plot(np.arange(audio.size) / sample_rate, audio, "b")
        ax1.set_xlim([-0.01, int(dur) + 1])
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.set_ylabel("Signal")
        ax1.set_ylim([-1, 1])

        pred = [1 if p > threshold else 0 for p in proba_s]
        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(len_pred) / (1 / FRAME_LEN), np.array(pred), "r", label="pred"
        )
        ax2.plot(
            np.arange(len_pred) / (1 / FRAME_LEN),
            np.array(proba_s),
            "g--",
            label="speech prob",
        )
        ax2.tick_params(axis="y", labelcolor="r")
        legend = ax2.legend(loc="lower right", shadow=True)
        ax1.set_ylabel("prediction")

        ax2.set_title(f"step {FRAME_LEN}s, buffer size {buffer_size}s")
        ax2.set_ylabel("Preds and Probas")
    ax = plt.subplot(num + 1, 1, i + 2)
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=sample_rate, fmax=8000
    )
    ax.set_title("Mel-frequency spectrogram")
    ax.grid()
    plt.show()


def main():
    file = "../../audio-classification/sns_classification/tests/resources/VAD_demo.wav"
    if not os.path.exists(file):
        os.system(
            'wget "https://dldata-public.s3.us-east-2.amazonaws.com/VAD_demo.wav" '
        )

    RATE = 16_000
    speech_array = load_and_resample(file, RATE)

    num_frames, sample_rate = torchaudio_info(file)
    audio, sample_rate = librosa.load(file, sr=sample_rate)
    dur = librosa.get_duration(audio)
    print(dur)

    threshold = 0.2
    STEP_LIST = [0.01]
    WINDOW_SIZE_LIST = [0.4]

    results = []
    for STEP, WINDOW_SIZE in zip(
        STEP_LIST,
        WINDOW_SIZE_LIST,
    ):
        arrays = list(break_into_chunks(speech_array, int(RATE * STEP)))

        vad = NeMoVAD(
            threshold=threshold,
            frame_duration=STEP,
            window_len_in_secs=WINDOW_SIZE,
        )

        print(f"====== STEP is {STEP}s, WINDOW_SIZE is {WINDOW_SIZE}s ====== ")
        preds, proba_b, proba_s = offline_inference(
            vad,
            arrays,
        )
        results.append([STEP, WINDOW_SIZE, preds, proba_b, proba_s])

    visualize(results, audio, sample_rate, threshold, dur)


if __name__ == "__main__":

    main()

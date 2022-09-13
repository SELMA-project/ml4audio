import os
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Iterator, Optional, Generator, List, Tuple, Any

import numpy as np

from misc_utils.buildable import Buildable
from ml4audio.audio_utils.torchaudio_utils import torchaudio_info
from nemo_vad.nemo_streaming_vad import NeMoVAD


@dataclass
class AudioChunk:
    id: str
    audio_array: np.ndarray


@dataclass
class VoiceSegment:
    array: np.ndarray = field(repr=False)  # int16 array
    start: str
    end: Optional[str] = None

    def is_final(self):
        return self.end is not None


TARGET_SAMPLE_RATE = 16_000  # fixed until trained own model on different sample_rate


def build_buffer_audio_arrays_generator(chunk_size=int(16000 * 0.1)) -> Generator:
    """
    TODO: oh-du-schreckliche!
        why did I ever wanted this to be a generator?
    """
    chunk = yield
    buffer = np.zeros(0, dtype=np.int16)
    while chunk is not None:
        valid_chunk: Optional[np.ndarray] = None
        assert buffer.dtype == chunk.dtype
        buffer = np.concatenate([buffer, chunk])
        if len(buffer) >= chunk_size:
            valid_chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
        chunk = yield valid_chunk

    if len(buffer) > 0:
        assert len(buffer) < chunk_size
        yield buffer
        # could be that part of the signal is thrown away


@dataclass
class StreamingSignalSegmentor(Buildable):
    vad: NeMoVAD
    # frame_dur: float = 0.1
    # input_sample_rate: int = TARGET_SAMPLE_RATE

    def _build_self(self) -> Any:
        self.segments_generator = None
        self.chunk_generator = None
        self.frame_dur = self.vad.frame_duration
        self.input_sample_rate = self.vad.input_sample_rate

    def handle_audio_array(self, array: np.ndarray) -> Optional[VoiceSegment]:
        assert array.dtype == np.int16
        assert len(array) <= self.frame_dur * self.input_sample_rate
        if self.chunk_generator is None:
            self.chunk_generator = build_buffer_audio_arrays_generator(
                chunk_size=int(self.frame_dur * self.input_sample_rate)
            )
            self.chunk_generator.send(None)
        audio_chunk: Optional[np.ndarray] = self.chunk_generator.send(array)
        if audio_chunk is not None:
            # audio_chunk = self.resampler(audio_chunk)
            seg = self.handle_valid_audio_chunk(
                AudioChunk(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}", audio_chunk
                )
            )
        else:
            seg = None
        return seg

    def handle_valid_audio_chunk(self, chunk: AudioChunk) -> Optional[VoiceSegment]:
        assert chunk.audio_array.shape[0] >= self.frame_dur * self.input_sample_rate
        if self.segments_generator is None:
            self.segments_generator = generate_segments_from_chunk(self.vad)
            self.segments_generator.send(None)
        return self.segments_generator.send(chunk)

    def flush(self) -> Optional[VoiceSegment]:
        if self.segments_generator is not None:
            assert self.chunk_generator is not None
            chunks = list(self.chunk_generator)  # flush audio-chunk buffer
            self.chunk_generator = None
            assert len(chunks) <= 1
            if len(chunks) == 1:
                chunk = AudioChunk(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}", chunks[0]
                )
                self.segments_generator.send(chunk)

            segments: List[VoiceSegment] = list(self.segments_generator)
            assert len(segments) <= 1
            if len(segments) == 1:
                assert segments[0].is_final()
                last_segment = segments[0]
            else:
                last_segment = None

            self.segments_generator = None
        else:
            last_segment = None
        return last_segment


def generate_segments_from_chunk(
    vad: NeMoVAD,
) -> Generator[Optional[VoiceSegment], AudioChunk, None]:

    buffer: List[VoiceSegment] = []
    no_voice_patience = 2
    memory_len = no_voice_patience // 2
    memory: List[Optional[VoiceSegment]] = [None for _ in range(memory_len)]
    remember = lambda: [m for m in memory if m is not None]

    patience = no_voice_patience
    is_ongoing_speech = lambda: len(buffer) > 0
    ac = yield
    ac: AudioChunk
    # TODO max buffer size? currently there is NO upper limit for segment lenght!

    while ac is not None:
        vs: Optional[VoiceSegment] = None
        is_voice = vad.is_speech(ac.audio_array)
        if is_voice or patience > 0:
            buffer += [VoiceSegment(ac.audio_array, ac.id)]
            vs = VoiceSegment(
                np.concatenate([s.array for s in buffer], dtype=np.int16),
                start=buffer[0].start,
            )

        elif not is_voice and is_ongoing_speech():
            assert len(buffer) > memory_len
            buffer = remember() + buffer[:-memory_len]
            completed_segment = VoiceSegment(
                np.concatenate([s.array for s in buffer], dtype=np.int16),
                start=buffer[0].start,
                end=buffer[-1].start,
            )
            vs = completed_segment
            buffer = []
            memory = memory[1:] + [VoiceSegment(ac.audio_array, ac.id)]
        elif not is_voice and not is_ongoing_speech():
            # no voice detected but also not ongoing speech
            memory = memory[1:] + [VoiceSegment(ac.audio_array, ac.id)]
        else:
            assert False

        if not is_voice:
            patience -= 1
        else:
            patience = no_voice_patience
        ac = yield vs
    else:
        if is_ongoing_speech():
            buffer = remember() + buffer
            completed_segment = VoiceSegment(
                np.concatenate([s.array for s in buffer], dtype=np.int16),
                buffer[0].start,
                buffer[-1].start,
            )
            yield completed_segment


if __name__ == "__main__":
    base_path = f"{os.environ['HOME']}"
    file = f"{base_path}/data/tmp/sample.wav"
    # os.system(f"play '{file}'")
    os.system(f"soxi '{file}'")

    frame_dur = 0.1
    RATE = 8000
    segmenter = StreamingSignalSegmentor(
        vad=NeMoVAD(
            vad_model_name="vad_marblenet",
            threshold=0.3,
            frame_duration=frame_dur,
            window_len_in_secs=0.5,
            # TODO: what is "good" window-len? below 1 sec feasible?
            input_sample_rate=RATE,
        ),
        frame_dur=frame_dur,
        input_sample_rate=RATE,
    )
    segmenter.init()
    num_frames, sample_rate, dur = torchaudio_info(file)
    voice_segs, dur = segmenter.speech_arrays_from_file(file)
    speech_arrays = [vs.array for vs in voice_segs]
    print(f"infer-speed: {(num_frames/sample_rate)/dur}")
    # array = np.concatenate(speech_arrays)
    # for a in speech_arrays:
    #     sleep(1)
    #     sf.write("/tmp/audio.wav", a, samplerate=RATE)
    #     os.system("play /tmp/audio.wav")
    # os.system("soxi /tmp/audio.wav")

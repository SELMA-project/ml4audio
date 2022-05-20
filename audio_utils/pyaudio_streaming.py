import wave
from typing import Iterator

CHUNK_SIZE = 1000


def build_pyaudio_stream(
    rate=16000, num_seconds_to_record=4, chunk_len=1.0
) -> Iterator[bytes]:
    import pyaudio

    chunk_size = 2 * round(chunk_len * rate)  # 16bit need 2 bytes
    pyaudio_object = pyaudio.PyAudio()
    stream: pyaudio.Stream = pyaudio_object.open(
        channels=1,
        format=pyaudio.paInt16,  # TODO
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    try:
        for _ in range(int(num_seconds_to_record * rate / chunk_size)):
            yield stream.read(chunk_size)
    finally:
        stream.close()
        pyaudio_object.terminate()


def pyaudio_play_stream_from_file(audio_file):
    import pyaudio

    with wave.open(audio_file, "rb") as wf:

        p = pyaudio.PyAudio()
        formatt = p.get_format_from_width(wf.getsampwidth())
        stream = p.open(
            format=formatt,
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        data = wf.readframes(CHUNK_SIZE)

        # play stream (3)
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(CHUNK_SIZE)

        # stop stream (4)
        stream.stop_stream()
        stream.close()

        # close PyAudio (5)
        p.terminate()

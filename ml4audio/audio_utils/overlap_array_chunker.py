from dataclasses import dataclass, field
from typing import (
    Iterator,
    Union,
    Iterable,
    Optional,
)

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from misc_utils.beartypes import Numpy1DArray
from misc_utils.utils import Singleton


@dataclass
class MessageChunk:
    """
    instance of this represents one chunks of an audio-message
    an audio-message can be split into possibly overlapping chunks, entire message got one message_id
    frame_idx is counter/absolut-position of audio-chunk's start frame in entire audio-message
    """

    message_id: str  # same for all chunks of same message
    frame_idx: int  # points to very first frame of this chunk
    array: NDArray
    end_of_signal: bool = False


@dataclass
class _DONT_EMIT_PREMATURE_CHUNKS(metaclass=Singleton):
    pass


DONT_EMIT_PREMATURE_CHUNKS = _DONT_EMIT_PREMATURE_CHUNKS()


@beartype
def messages_from_chunks(
    signal_id: str, chunks: Iterable[NDArray]
) -> Iterator[MessageChunk]:
    frame_idx = 0
    dtype = None
    for chunk in chunks:
        if dtype is None:
            dtype = chunk.dtype
        yield MessageChunk(message_id=signal_id, frame_idx=frame_idx, array=chunk)
        frame_idx += len(chunk)

    len_of_dummy_chunk = (
        0  # TODO does empty dummy-chunk really not break anything downstream?
    )
    shape = [x for x in chunk.shape]
    shape[0] = len_of_dummy_chunk
    dummy_chunk_just_to_transport_eos = np.zeros(shape, dtype=dtype)
    yield MessageChunk(
        message_id=signal_id,
        frame_idx=frame_idx,
        array=dummy_chunk_just_to_transport_eos,
        end_of_signal=True,
    )


@dataclass
class OverlapArrayChunker:
    """
    TODO: this is a variable step-size sequencer chunker! -> split into two classes: 1. "fixed" and 2. "variable" chunk-size

    call it "SegChunker" ? in the end its sequences that are being chunked!
    formerly called AudioMessageChunker
    TODO: why is this not buildable? where build_self essentially calls reset
    does chunking
    input-stream: consists of numpy arrays
    output-stream: consists of (chunked) numpy-arrays which are longer than minimum_chunk_size and at most "chunk_size" long
    indepentently of chunk_size of input-stream an output-chunk is yielded every "step_size"
    at very beginning of input-stream "premature-chunks" are yielded (every step-size)
    after internal buffer grew bigger than chunk_size, it behaves as ring-buffer and further output_chunks all (but very last) have chunk_size
    at very end it is flushed what remained in buffer
    """

    chunk_size: int
    min_step_size: int  # if step_size==chunk_size it produced non-overlapping segments
    _buffer: Optional[Numpy1DArray] = field(init=False, repr=False, default=None)
    minimum_chunk_size: Union[
        int, _DONT_EMIT_PREMATURE_CHUNKS
    ] = DONT_EMIT_PREMATURE_CHUNKS
    max_step_size: Optional[int] = None

    frame_counter: Optional[int] = field(init=False, repr=False, default=None)
    last_buffer_size: int = field(
        init=False, repr=False, default=None
    )  # default=None forces need to reset

    def reset(self) -> None:
        self._buffer = None
        self.frame_counter = None
        self.last_buffer_size = 0

    @property
    def _buffer_size(self) -> int:
        return self._buffer.shape[0] if self._buffer is not None else 0

    def __can_yield_full_grown_chunk(self):
        if self.is_very_start:
            return self._buffer_size >= self.chunk_size
        else:
            return self._buffer_size >= self.chunk_size + self.min_step_size

    def __premature_chunk_long_enough_to_yield_again(self):
        """
        if premature-chunk grew bigger by step-size compared to last time it was yielded
        """
        return (
            self._buffer_size >= self.last_buffer_size + self.min_step_size
        )  # alter!!

    @beartype
    def _calc_step_size(self, buffer_len: int, is_very_start: bool) -> int:

        if self.max_step_size is None:
            sz = self.min_step_size
        else:
            sz = min(self.max_step_size, buffer_len - self.chunk_size)
        return sz if not is_very_start else 0

    @property
    def is_very_start(self):
        return self.frame_counter is None

    @beartype
    def handle_datum(self, datum: MessageChunk) -> list[MessageChunk]:
        self._check_framecounter_consistency(datum)

        self._buffer = (
            np.concatenate([self._buffer, datum.array], axis=0)
            if self._buffer is not None
            else datum.array
        )
        if self.__can_yield_full_grown_chunk():
            output_messages = self._fullgrown_chunks(
                datum,
            )
            print(f"{output_messages=}")

        elif (
            self.minimum_chunk_size is not DONT_EMIT_PREMATURE_CHUNKS
            and self._buffer_size >= self.minimum_chunk_size
            and self.is_very_start
            and self.__premature_chunk_long_enough_to_yield_again()
        ):
            self.last_buffer_size = self._buffer_size
            premature_chunk = self._buffer
            output_messages = [
                MessageChunk(
                    message_id=datum.message_id,
                    array=premature_chunk,
                    frame_idx=0,
                    end_of_signal=datum.end_of_signal,  # can happen for short audio-signals!
                )
            ]
        else:
            output_messages = []
        if len(output_messages) > 0:
            assert sum(1 for om in output_messages if om.end_of_signal) <= 1
            yielded_final = output_messages[-1].end_of_signal
        else:
            yielded_final = False

        if datum.end_of_signal:
            if not yielded_final:
                assert self._buffer_size > 0
                output_messages += [self._do_flush(datum.message_id)]
            self.reset()

        return output_messages

    def _check_framecounter_consistency(self, datum):
        if not self.is_very_start:
            if self.frame_counter + self._buffer_size != datum.frame_idx:
                assert (
                    False
                ), f"frame-counter inconsistency: {self.frame_counter + self._buffer_size=} != {datum.frame_idx=}"

    def _fullgrown_chunks(self, datum: MessageChunk) -> list[MessageChunk]:
        msg_chunks = []
        while self.__can_yield_full_grown_chunk():
            step_size = self._calc_step_size(len(self._buffer), self.is_very_start)
            self._buffer = self._buffer[step_size:]
            self.frame_counter = (
                self.frame_counter + step_size if not self.is_very_start else step_size
            )

            full_grown_chunk = self._buffer[: self.chunk_size]

            msg_chunks.append(
                MessageChunk(
                    message_id=datum.message_id,
                    array=full_grown_chunk,
                    frame_idx=self.frame_counter,
                    end_of_signal=datum.end_of_signal
                    and len(self._buffer) == self.chunk_size,
                )
            )
        return msg_chunks

    @beartype
    def _do_flush(self, message_id: str) -> MessageChunk:
        assert (
            self._buffer_size <= self.chunk_size + self.min_step_size
        ), f"cannot happen that len of buffer: {self._buffer_size} > {self.chunk_size=}"
        last_step_size = max(0, self._buffer_size - self.chunk_size)
        flushed_chunk = self._buffer[-self.chunk_size :]
        last_frame_count = self.frame_counter if self.frame_counter is not None else 0
        frame_idx = last_frame_count + last_step_size
        return MessageChunk(
            message_id=message_id,
            array=flushed_chunk,
            frame_idx=frame_idx,
            end_of_signal=True,
        )

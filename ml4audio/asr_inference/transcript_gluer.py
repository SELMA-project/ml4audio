import difflib
import os
from dataclasses import field, dataclass
from typing import Iterator, Tuple, AsyncIterable, AsyncIterator
from typing import Optional

from beartype import beartype

from misc_utils.buildable import Buildable
from ml4audio.asr_inference.asr_array_stream_inference import ASRMessage
from ml4audio.asr_inference.transcript_glueing import (
    glue_left_right_update_hyp_buffer,
    NonEmptyAlignedTranscript,
)
from ml4audio.audio_utils.aligned_transcript import (
    AlignedTranscript,
)

DEBUG = os.environ.get("DEBUG", "False").lower() != "false"
if DEBUG:
    print("TranscriptGluer DEBUGGING MODE")


@dataclass
class ASRStreamInferenceOutput:
    id: str
    ending_to_be_removed: str  # end of transcript that should be removed
    text: str
    aligned_transcript: AlignedTranscript  # TODO: why WAS this optional?
    end_of_message: bool = False


@dataclass
class TranscriptGluer(Buildable):
    """
    ───▄▄▄
    ─▄▀░▄░▀▄
    ─█░█▄▀░█
    ─█░▀▄▄▀█▄█▄▀
    ▄▄█▄▄▄▄███▀

    """

    _hyp_buffer: Optional[NonEmptyAlignedTranscript] = field(
        init=False, repr=False, default=None
    )
    seqmatcher: Optional[difflib.SequenceMatcher] = field(
        init=False, repr=False, default=None
    )

    def __enter__(self):
        return self.build()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def reset(self) -> None:
        self._hyp_buffer: Optional[NonEmptyAlignedTranscript] = None

    def _build_self(self):
        self.reset()
        self.seqmatcher = difflib.SequenceMatcher()

    @beartype
    def handle_message(self, inp: ASRMessage) -> ASRStreamInferenceOutput:
        suffix_to_be_removed, new_suffix = self._calc_suffix(inp)
        output = ASRStreamInferenceOutput(
            id=inp.message_id,
            ending_to_be_removed=suffix_to_be_removed,
            text=new_suffix,
            aligned_transcript=self._hyp_buffer,
            end_of_message=inp.end_of_message,
        )
        if DEBUG:
            print(
                f"{inp.aligned_transcript.text=}, {suffix_to_be_removed=},{new_suffix=},{self._hyp_buffer.text=}"
            )

        assert self._hyp_buffer is not None
        # if inp.end_of_message: TODO: resetting here breaks the AsrStreamingPostASRSegmentingExecutor
        #     self.reset()
        return output

    @beartype
    def _calc_suffix(self, inp: ASRMessage) -> tuple[str, str]:

        if (
            inp.aligned_transcript.len_not_space > 0
        ):  # TODO: message can be fine and end_of_message at same time!
            is_very_start_of_stream = inp.aligned_transcript.frame_id == 0
            do_overwrite_everything = (
                is_very_start_of_stream or self._hyp_buffer is None
            )
            if do_overwrite_everything:
                ending_to_be_removed = self._get_ending_to_be_removed(
                    self._hyp_buffer, inp
                )

                self._hyp_buffer = inp.aligned_transcript

                text = inp.aligned_transcript.text
                new_suffix = (
                    " " + text
                    if not is_very_start_of_stream and len(text.replace(" ", "")) > 0
                    else text
                )
            else:
                assert self._hyp_buffer is not None
                # if detect_offset_shift(
                #     earlier=al_tr.aligned_transcript.offset, later=self.hyp_buffer.offset
                # ):
                #     raise NotImplementedError("TODO!!!")
                #     self.hyp_buffer.offset -= OFFSET_SHIFT
                #     print("offset-shift has happended!")
                (
                    ending_to_be_removed,
                    text,
                    hyp_buffer,
                ) = glue_left_right_update_hyp_buffer(
                    inp.aligned_transcript, self._hyp_buffer, self.seqmatcher
                )
                self._hyp_buffer = hyp_buffer
                new_suffix = f"{text}{' ' if inp.end_of_message else ''}"

        elif inp.end_of_message:
            if DEBUG:
                print("zero message just for end_of_message flag")
            ending_to_be_removed = ""
            new_suffix = ""
        else:
            if DEBUG:
                print(f"NOT fine and NOT end_of_message!!")
            ending_to_be_removed = ""
            new_suffix = ""

        input_not_fine_but_final = self._hyp_buffer is None
        if input_not_fine_but_final:
            self._hyp_buffer = inp.aligned_transcript
        return ending_to_be_removed, new_suffix

    def _get_ending_to_be_removed(self, hyp_buffer, inp):
        if hyp_buffer is not None:
            ending_to_be_removed = hyp_buffer.text
            assert len(ending_to_be_removed) > 0, inp.aligned_transcript.text
        else:
            ending_to_be_removed = ""
        return ending_to_be_removed

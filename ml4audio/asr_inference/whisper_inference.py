from dataclasses import dataclass
from typing import Annotated, Optional, Union

from beartype import beartype
from beartype.vale import Is

from ml4audio.audio_utils.audio_segmentation_utils import StartEnd


@beartype
def fix_start_end(seg: dict, audio_dur: float) -> StartEnd:
    start = seg["start"]
    end = seg["end"]
    if start < 0:
        print(f"WTF! whisper gave {start=}")
        start = 0.0

    if end > audio_dur:
        fixed_end = min(audio_dur, end)
        print(f"WTF! whisper gave {end=} that is after {audio_dur=} -> {fixed_end=}")
        end = fixed_end

    if end - start <= 0.08:
        print(f"WTF! whisper gave {(start,end)=}")
        start = end - 0.04
        end = min(audio_dur, start + 0.04)

    return (start, end)


@dataclass
class WhisperArgs:
    task: Annotated[str, Is[lambda s: s in WHISPER_TASKS]]
    language: str = "de"
    temperature: Optional[Union[float, tuple[float, ...], list[float]]] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )  # this is default in whisper code
    # don't mess with the temperatures! they are needed for fallback if beam-search fails!
    beam_size: Optional[int] = None  # default=5 see whisper code


WHISPER_TASKS = {"transcribe", "translate"}

import re
from dataclasses import dataclass

from beartype import beartype
from webvtt import WebVTT, Caption

from misc_utils.dataclass_utils import deserialize_dataclass
from misc_utils.utils import iterable_to_chunks
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript, LetterIdx

try:
    from pysubs2.time import make_time
    from pysubs2 import SSAFile, Color, SSAEvent
except ImportError:
    pass

TARGET_SAMPLE_RATE = 16_000


@dataclass
class SubtitleBlock:
    start: int  # in ms
    end: int  # in ms
    name_texts: list[tuple[str, str]]

    @property
    def names(self):
        return [n for n, _ in self.name_texts]

    # TODO(tilo): who wants this from_dict_letters?
    @classmethod
    def from_dict_letters(cls, dictletter: dict[str, list[LetterIdx]]):
        first_index = list(dictletter.values())[0][0].r_idx
        start = make_time(ms=round(1000 * first_index / TARGET_SAMPLE_RATE))
        last_index = list(dictletter.values())[0][-1].r_idx
        end = make_time(ms=round(1000 * last_index / TARGET_SAMPLE_RATE))
        return cls(
            start,
            end,
            [
                (name, "".join((l.letter for l in letters)))
                for name, letters in dictletter.items()
            ],
        )


@dataclass
class StyleConfig:
    fontsize: float = 20.0


@beartype
def create_ass_file(
    subtitle_blocks: list[SubtitleBlock], ass_file: str, styles: dict[str, StyleConfig]
):
    subs = SSAFile()
    colors = [Color(255, 255, 255), Color(0, 255, 255), Color(255, 255, 100)]
    for k, name in enumerate(subtitle_blocks[0].names):
        my_style = subs.styles["Default"].copy()
        my_style.primarycolor = colors[k]
        my_style.fontsize = styles[name].fontsize
        my_style.shadow = 0
        subs.styles[name] = my_style

    for sb in subtitle_blocks:
        start, end = None, None
        for name, text in sb.name_texts:
            if len(text) > 0:
                text = text.replace("_", " ")
                if start is None:
                    start = sb.start
                    end = sb.end
                sub_line = SSAEvent(
                    start=start,
                    end=end,
                    text=text,
                )
                sub_line.style = name
                subs.append(sub_line)
            else:
                print(f"WARNING: got empty block! {name} ")
    subs.save(ass_file)


def ms_to_HMSf(ms):
    """
    stolen from youknowwhere/sandbox/hugging_face_asr_experiment/-/blob/master/convert_to_webvtt.py#L5
    """
    hours = ms // 3600000
    ms_m = ms - hours * 3600000
    minutes = ms_m // 60000
    ms_s = ms_m - minutes * 60000
    seconds = ms_s // 1000
    ms_ms = ms_s - seconds * 1000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms_ms:03d}"  # modified this line


@dataclass
class TranslatedSubtitleBlock:
    text: str
    translation: str
    start: float
    end: float


@beartype
def write_webvtt(
    blocks: list[TranslatedSubtitleBlock],
    vtt_file: str,
    get_subtitles=lambda x: [x.text, x.translation],
):
    vtt = WebVTT()
    for block in blocks:
        caption = Caption(
            ms_to_HMSf(round(block.start * 1000)),
            ms_to_HMSf(round(block.end * 1000)),
            get_subtitles(block),
        )
        vtt.captions.append(caption)
    vtt.save(vtt_file)


@beartype
def write_webvtt_file(
    start_end_text: list[tuple[float, float, str]],
    vtt_file: str,
):
    vtt = WebVTT()
    for s, e, t in start_end_text:
        caption = Caption(
            ms_to_HMSf(round(s * 1000)),
            ms_to_HMSf(round(e * 1000)),
            t,
        )
        vtt.captions.append(caption)
    vtt.save(vtt_file)


TimeStampledTokens = list[tuple[str, float]]


@beartype
def time_slice(
    token_timestamps: list[tuple[str, float]], start: float, end: float
) -> list[str]:
    return [token for token, ts in token_timestamps if ts >= start and ts < end]


def remove_to_many_spaces(s: str) -> str:
    return re.sub("\s+", " ", s)


@beartype
def calc_subtitles_blocks_given_segments(
    ts_tokens: TimeStampledTokens,
    tts_tokens: TimeStampledTokens,
    start_ends: list[tuple[float, float]],
) -> list[TranslatedSubtitleBlock]:
    subtitle_blocks = []
    for k, (start, end) in enumerate(start_ends):
        text = remove_to_many_spaces(" ".join(time_slice(ts_tokens, start, end)))
        translation = remove_to_many_spaces(
            " ".join(time_slice(tts_tokens, start, end))
        )
        subtitle_blocks.append(
            TranslatedSubtitleBlock(
                text=text,
                translation=translation,
                start=start,
                end=end,
            )
        )
    return subtitle_blocks


@beartype
def calc_subtitles_blocks(
    ts_tokens: TimeStampledTokens,
    tts_tokens: TimeStampledTokens,
    max_num_characters: int = 50,
) -> list[TranslatedSubtitleBlock]:

    chunks = list(
        iterable_to_chunks(
            ts_tokens,
            is_yieldable_chunk=lambda b: sum([len(t) for t, s in b])
            > max_num_characters,
        )
    )
    subtitle_blocks = []
    for k, chunk in enumerate(chunks):
        chunk: list[tuple[str, float]]
        _, start = chunk[0]
        _, end = chunks[k + 1][0] if k < len(chunks) - 1 else ("nix", 10000.0)
        text = remove_to_many_spaces(" ".join([t for t, s in chunk]))
        translated_tokens_within_time_range = [
            t for t, s in tts_tokens if s >= start and s < end
        ]
        translation = remove_to_many_spaces(
            " ".join(translated_tokens_within_time_range)
        )
        subtitle_blocks.append(
            TranslatedSubtitleBlock(
                text=text,
                translation=translation,
                start=start,
                end=end,
            )
        )
    return subtitle_blocks


@beartype
def lininterpolate_start_end(num_steps: int, start: float, end: float) -> list[float]:
    start, end = round(start, 3), round(end, 3)

    if num_steps > 1:
        tokens_ts = [
            round(start + (end - start) * k / (num_steps - 1), 3)
            for k in range(num_steps)
        ]
        assert (
            tokens_ts[0] == start and tokens_ts[-1] == end
        ), f"{tokens_ts[0]=},{tokens_ts[-1]=},{start=},{end=}"
    else:
        tokens_ts = [start]
    return tokens_ts


@beartype
def calc_time_stamped_tokens(
    data: list[dict],
) -> tuple[TimeStampledTokens, TimeStampledTokens]:
    ts_tokens, tts_tokens = [], []
    for d in data:
        altr_ru: AlignedTranscript = deserialize_dataclass(d["text_punctcap"])
        altr_en: AlignedTranscript = deserialize_dataclass(d["translation_ru-en"])
        tokens = altr_ru.text.split(" ")
        start = d["start"] / altr_ru.sample_rate  # altr_ru.abs_timestamps[0]
        end = d["end"] / altr_ru.sample_rate  # altr_ru.abs_timestamps[-1]
        ts = lininterpolate_start_end(len(tokens), start, end)
        ts_tokens.extend([(t, s) for t, s in zip(tokens, ts)] + [(" ", end)])

        ttokens = altr_en.text.split(" ")
        # print(f"{tokens=}\n{ttokens=}")
        tts = lininterpolate_start_end(len(ttokens), start, end)
        tts_tokens.extend([(t, s) for t, s in zip(ttokens, tts)] + [(" ", end)])
    assert len(ts_tokens) > 0
    return ts_tokens, tts_tokens

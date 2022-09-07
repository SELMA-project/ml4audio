from dataclasses import dataclass
from typing import List, Tuple, Dict

try:
    from pysubs2 import SSAFile, Color, SSAEvent
except ImportError:
    pass

@dataclass
class SubtitleBlock:
    start: int  # in ms
    end: int  # in ms
    name_texts: list[Tuple[str, str]]

    @property
    def names(self):
        return [n for n, _ in self.name_texts]


@dataclass
class StyleConfig:
    fontsize: float = 20.0


def create_ass_file(
    subtitle_blocks: list[SubtitleBlock], ass_file, styles: dict[str, StyleConfig]
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

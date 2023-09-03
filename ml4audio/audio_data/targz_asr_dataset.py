import itertools
import os
import tarfile
from abc import abstractmethod
from dataclasses import field, dataclass
from pathlib import Path
from typing import Iterator, Optional, ClassVar, Union, Iterable, Any

from beartype import beartype
from tqdm import tqdm

from data_io.readwrite_files import (
    filter_gen_targz_members,
    write_jsonl,
    read_jsonl,
)
from misc_utils.beartypes import NeStr, bearify
from misc_utils.buildable import Buildable
from misc_utils.buildable_data import BuildableData
from misc_utils.cached_data import (
    CachedData,
)
from misc_utils.dataclass_utils import (
    UNDEFINED,
    _UNDEFINED,
)
from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix
from misc_utils.utils import TimedIterable, just_try
from ml4audio.audio_utils.audio_data_models import (
    AudioTextData,
    ArrayText,
    FileLikeAudioCorpus,
    SegmentCorpus,
    SegmentAnnotation,
)
from ml4audio.audio_utils.audio_io import (
    FileLikeAudioDatum,
    load_audio_array_from_filelike,
)


@dataclass
class TarGzTranscripts(BuildableData):
    targz_file: str = "some-tar.gz file"
    split_names: ClassVar[list[str]] = ["train", "dev", "test"]
    base_dir: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["raw_data"])
    split2id2transcript: dict[str, dict[str, str]] = field(
        init=False, repr=False, default=UNDEFINED
    )

    @property
    def _is_data_valid(self) -> bool:
        return all(
            os.path.isfile(f"{self.data_dir}/{split}.jsonl")
            for split in self.split_names
        )

    @property
    def name(self):
        targz_name = Path(self.targz_file).stem.replace(".tar", "")
        return f"{targz_name}"

    @beartype
    def extract_transcript_files(self) -> Iterator[str]:
        for member, f in tqdm(
            filter_gen_targz_members(self.targz_file, self.contains_transcript),
            desc=f"extract_transcript_files from {self.targz_file}",
        ):
            file_path = (
                f"{self.extract_folder}/{self.build_transcript_file_name(member.name)}"
            )
            os.makedirs(str(Path(file_path).parent), exist_ok=True)
            with open(file_path, mode="wb") as wf:
                wf.write(f.read())
                yield file_path
                if self.found_all_transcripts(file_path):
                    print(f"found all transcripts-files!")
                    break

    def _build_data(self) -> Any:
        self.extract_folder = (
            f"{self.data_dir}/raw_extracts"  # tilo: this could actually be tmp-dir!
        )
        os.makedirs(self.extract_folder, exist_ok=True)

        transcript_files = list(self.extract_transcript_files())
        assert len(transcript_files) == len(
            self.split_names
        ), f"did not find transcript-files!"

        for split_name in self.split_names:
            write_jsonl(
                f"{self.data_dir}/{split_name}.jsonl",
                (
                    {"id": eid, "text": text}
                    for eid, text in self.build_id_transcripts(
                        split_name, transcript_files
                    )
                ),
            )
        self._load_data()

    def _load_data(self) -> None:
        self.split2id2transcript = {
            split: {
                d["id"]: d["text"] for d in read_jsonl(f"{self.data_dir}/{split}.jsonl")
            }
            for split in self.split_names
        }

    @abstractmethod
    def contains_transcript(self, member: tarfile.TarInfo) -> bool:
        raise NotImplementedError

    @beartype
    def build_transcript_file_name(self, member_name: str) -> str:
        return member_name

    @beartype
    @abstractmethod
    def build_id_transcripts(
        self, split_name: str, transcript_files: list[str]
    ) -> list[tuple[str, str]]:
        raise NotImplementedError

    @beartype
    def found_all_transcripts(self, file_path: str) -> bool:
        if not hasattr(self, "todo"):
            self.todo = [x for x in self.split_names]

        for s in self.split_names:
            if file_path.endswith(f"{s}/transcripts.txt"):
                self.todo.pop(self.todo.index(s))
        return len(self.todo) == 0


@dataclass
class TranscribedAudio:
    """
    TODO: does not carry segmenation information! offset+duraction / start end assumes that entire audio is used!
    """

    audio_datum: FileLikeAudioDatum
    text: NeStr  # am I too strict here? no, cause even if audio contains just noise, at least a space " " should be there jiwer has this:  raise ValueError("one or more groundtruths are empty strings")


@dataclass
class TranscribedAudioCorpus(Iterable[TranscribedAudio]):
    @property
    def id(self) -> NeStr:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[TranscribedAudio]:
        raise NotImplementedError


@dataclass
class TarGzASRCorpus(TranscribedAudioCorpus, Buildable):
    """
    TODO: better naming!
    """

    targztranscripts: Union[_UNDEFINED, TarGzTranscripts] = UNDEFINED
    split: str = "dev"

    def _build_self(self) -> Any:
        pass

    @property
    def id(self):
        return self.name

    @property
    def name(self):
        targz_name = Path(self.targztranscripts.targz_file).stem.replace(".tar", "")
        return f"{targz_name}-{self.split}"

    @property
    def id2transcript(self):
        return self.targztranscripts.split2id2transcript[self.split]

    @abstractmethod
    def audiofile_to_id(self, member_name: str) -> str:
        """
        here transcript-id is same as audio-id
        """
        raise NotImplementedError

    @abstractmethod
    def is_audiofile(self, member_name: str) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[TranscribedAudio]:
        assert self._is_ready
        assert self.targztranscripts.split2id2transcript is not UNDEFINED

        def is_audio_file_of_this_split(member: tarfile.TarInfo):
            return self.is_audiofile(member.name) and (
                self.audiofile_to_id(member.name) in self.id2transcript.keys()
            )

        for m, f in filter_gen_targz_members(
            self.targztranscripts.targz_file,
            is_audio_file_of_this_split,
            verbose=True,
        ):

            eid = self.audiofile_to_id(m.name)  # transcript-id equals audio-id!!
            text = self.id2transcript[eid]
            format = m.name.split(".")[-1]

            yield TranscribedAudio(
                audio_datum=FileLikeAudioDatum(id=eid, audio_source=f, format=format),
                text=text,
            )


@dataclass
class TarGzArrayText(AudioTextData, Buildable):
    """
    # TODO: how was this working without being buildable?
    rename to  ArrayTextFromTarGzASRCorpus ??
    actually the read-speed is less interesting, bottle neck comes after, when loading/resampling the audio
    """

    corpus: Union[_UNDEFINED, TarGzASRCorpus] = UNDEFINED
    sample_rate: int = 16_000
    limit: Optional[int] = None

    @property
    def name(self) -> str:
        return self.corpus.name

    def generate_raw_data(self):
        it = TimedIterable(self.corpus)
        for k, datum in enumerate(it):
            datum: TranscribedAudio
            eid = datum.audio_datum.id

            if k % 1000 == 0:
                print(f"read {k} samples from {self.name}, read-speed: {it.speed}")
            # TODO: what about segments? start-ends? transcripts which only span a part of the audio?
            array = just_try(
                lambda: load_audio_array_from_filelike(
                    datum.audio_datum,
                    self.sample_rate,
                ),
                default=None,
                verbose=True,
            )
            if array is not None:
                yield eid, array, datum.text
            else:
                print(f"failed to load {eid=},{format=}")

    def __iter__(self) -> Iterator[ArrayText]:
        for eid, array, text in itertools.islice(
            self.generate_raw_data(), 0, self.limit
        ):
            yield array, text


@dataclass
class TarGzArrayTextWithSize(TarGzArrayText, CachedData):
    """
    some-how looping over zipped corpus takes forever!
    calculating corpus size: 7409it [17:06:14,  9.56s/it]at position 110000 in cv-corpus-10.0-2022-07-04-es.tar.gz
    """

    cache_base: PrefixSuffix = field(default_factory=lambda: BASE_PATHES["raw_data"])
    use_hash_suffix: ClassVar[bool] = False
    size_in_hours: float = field(init=False, default=UNDEFINED)

    def _build_cache(self):
        g = (
            len(a) / self.sample_rate for a, t in tqdm(self, "calculating corpus size")
        )
        self.size_in_hours = sum(g) / (60 ** 2)


@dataclass
class TarGzAudioFileCorpus(FileLikeAudioCorpus, Buildable):
    """
    closely coupled to TarGzASRCorpus, cannot be circumvented here!
    """

    targz_corpus: Union[_UNDEFINED, TarGzASRCorpus] = UNDEFINED
    limit: Optional[int] = field(default=None, repr=False)

    def __iter__(self) -> Iterator[FileLikeAudioDatum]:
        g: Iterable[TranscribedAudio] = self.targz_corpus
        yield from (
            FileLikeAudioDatum(
                id=a.audio_datum.id,
                audio_source=a.audio_datum.audio_source,
                format=a.audio_datum.format,
            )
            for a in itertools.islice(g, 0, self.limit)
        )


# TODO what was this SegmentsFromTarGzASRCorpus good for?
@dataclass
class SegmentsFromTarGzASRCorpus(SegmentCorpus, Buildable):
    corpus: Union[_UNDEFINED, TarGzASRCorpus] = UNDEFINED

    def __post_init__(self):
        self.id = bearify(self.build_segment_corpus_id(self.corpus.name), NeStr)
        self.audiocorpus_id = bearify(self.corpus.name, NeStr)
        super().__post_init__()

    @classmethod
    def build_segment_corpus_id(cls, corpus_id):
        return f"{corpus_id}-segmentation"

    def __iter__(self) -> Iterator[SegmentAnnotation]:
        for eid in self.corpus.id2transcript.keys():
            yield SegmentAnnotation(id=eid, audio_id=eid)


# TODO: why would I ever need this extract thing?
# @dataclass
# class TarGzExtractASRDataset(ASRCorpus):
#     targz_corpus: TarGzASRCorpus = field(default=None)
#     sample_rate: int = 16000
#
#     @property
#     def wav_dir(self):
#         return self.prefix_cache_dir("wavs")
#
#     def get_audio_filepath(self, file: str) -> str:
#         file_name = file.replace(self.cache_dir, "")
#         return f"{self.cache_dir}/wavs/{file_name}"
#
#     def _gen_samples(self) -> Iterator[ASRSample]:
#         g: Iterator[tuple[Optional[NpFloatDim1], str]] = (
#             (try_to_read_audio(b, target_sample_rate=self.sample_rate), t)
#             for b, t in self.targz_corpus
#         )
#         os.makedirs(self.wav_dir)
#
#         def process(k, a, t) -> ASRSample:
#             audio_filepath = self.get_audio_filepath(f"{k}.wav")
#             sr = self.sample_rate
#             sf.write(audio_filepath, a, sr)
#             return ASRSample(
#                 id="k", audio_filepath=audio_filepath, sample_rate=sr, text=t
#             )
#
#         yield from (process(k, a, t) for k, (a, t) in enumerate(g) if a is not None)

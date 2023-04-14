import os.path
from dataclasses import dataclass, field
from typing import Any

from beartype import beartype
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from misc_utils.beartypes import NeList, NumpyFloat1D
from misc_utils.buildable_data import SlugStr
from misc_utils.dataclass_utils import UNDEFINED
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from ml4audio.asr_inference.inference import (
    ASRAudioSegmentInferencer,
    StartEndTextsNonOverlap,
)
from ml4audio.asr_inference.whisper_inference import (
    fix_start_end,
    WhisperArgs,
    WhisperInferencer,
)
from ml4audio.audio_utils.audio_io import ffmpeg_load_trim
from ml4audio.audio_utils.audio_segmentation_utils import (
    fix_segments_to_non_overlapping,
)


@dataclass
class HfPipelineWhisperASRSegmentInferencer(WhisperInferencer):

    model_name: str = "openai/whisper-base"
    chunk_length_s: float = 30.0  # see: https://huggingface.co/spaces/openai/whisper/discussions/67#63eb6ec2c6beb750e2cf47e9
    num_beams: int = 5
    hf_pipeline: AutomaticSpeechRecognitionPipeline = field(init=False, repr=False)

    base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "MODELS/WHISPER_MODELS")
    )

    @property
    def name(self) -> SlugStr:
        return f"hf-pipeline-{self.model_name}"

    @property
    def _is_data_valid(self) -> bool:
        """
        if paranoid one could check that all files exist

        added_tokens.json  generation_config.json  normalizer.json           pytorch_model.bin        tokenizer_config.json  vocab.json
        config.json        merges.txt              preprocessor_config.json  special_tokens_map.json  tokenizer.json

        """
        return os.path.isfile(f"{self.data_dir}/pytorch_model.bin")

    def _build_data(self) -> Any:
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            chunk_length_s=self.chunk_length_s
            # , device=args.device
        )
        whisper_pipeline.save_pretrained(self.data_dir)

    @beartype
    def parse_whisper_segments(
        self, whisper_segments: NeList[dict], audio_dur: float
    ) -> StartEndTextsNonOverlap:

        start_end = [fix_start_end(seg, audio_dur) for seg in whisper_segments]
        start_end = fix_segments_to_non_overlapping(start_end)
        return [
            (start, end, seg["text"])
            for seg, (start, end) in zip(whisper_segments, start_end)
        ]

    def __enter__(self):
        self._load_and_prepare_hf_pipeline()

    def _load_and_prepare_hf_pipeline(self):
        """
        see: https://github.com/huggingface/community-events/blob/606f62d19c818cde6331f809a37dbbab2dca85a8/whisper-fine-tuning-event/run_eval_whisper_streaming.py#L51
        """
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.data_dir,
            chunk_length_s=self.chunk_length_s
            # , device=args.device
        )
        self.hf_pipeline = whisper_pipeline  # noqa

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        del self.hf_pipeline

    def predict_transcribed_with_whisper_args(
        self, audio_array: NumpyFloat1D, whisper_args: WhisperArgs
    ) -> StartEndTextsNonOverlap:
        hfp = self.hf_pipeline
        whisper_prompt_ids = hfp.tokenizer.get_decoder_prompt_ids(
            language=whisper_args.language, task=whisper_args.task
        )
        hfp.model.config.forced_decoder_ids = whisper_prompt_ids
        # print(f"{whisper_pipeline.model.config.max_length=}")
        # NB: decoding option
        # limit the maximum number of generated tokens to 225
        # whisper_pipeline.model.config.max_length = 225 + 1
        # sampling
        # pipe.model.config.do_sample = True
        # beam search
        hfp.model.config.num_beams = self.num_beams
        # return
        hfp.model.config.return_dict_in_generate = True  # TODO: why?
        # pipe.model.config.output_scores = True
        # pipe.model.config.num_return_sequences = 5

        # audio_dur = float(len(audio_array) / self.sample_rate)
        output = self.hf_pipeline(audio_array, return_timestamps=True)
        return [
            (start, end, o["text"])
            for o in output["chunks"]
            for start, end in [o["timestamp"]]
        ]


@dataclass
class HfWhisperASRSegmentInferencer(ASRAudioSegmentInferencer):

    model_name: str
    whisper_args: WhisperArgs = UNDEFINED

    model: WhisperForConditionalGeneration = field(init=False, repr=False)
    processor: WhisperProcessor = field(init=False, repr=False)

    def __enter__(self):
        self._load_and_prepare()

    def _load_and_prepare(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name
        )  # .to(device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            language=self.whisper_args.language,
            task=self.whisper_args.task,
        )
        # NB: set forced_decoder_ids for generation utils -> das is auch scheisse!
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.whisper_args.language, task=self.whisper_args.task
        )

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        del self.model
        del self.processor

    @beartype
    def predict_transcribed_segments(
        self, audio_array: NumpyFloat1D
    ) -> StartEndTextsNonOverlap:
        model_sample_rate = self.processor.feature_extractor.sampling_rate

        inputs = self.processor(
            audio_array, sampling_rate=model_sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features
        # input_features = input_features.to(device)

        generated_ids = self.model.generate(
            inputs=input_features,
            max_new_tokens=225,  # TODO(tilo): WTF? see: https://huggingface.co/bofenghuang/whisper-large-v2-cv11-german
            return_timestamps=True,
        )  # greedy
        # generated_ids = model.generate(inputs=input_features, max_new_tokens=225, num_beams=5)  # beam search

        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, output_offsets=True
        )[0]
        return [
            (start, end, o["text"])
            for o in output["offsets"]
            for start, end in [o["timestamp"]]
        ]


if __name__ == "__main__":
    BASE_PATHES["cache_root"] = "/tmp/cache_root"
    """
    https://discuss.huggingface.co/t/support-for-asr-inference-on-longer-audiofiles-or-on-live-transcription/30464

    """
    file = "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"
    array = ffmpeg_load_trim(file, sr=16000)

    asr = HfPipelineWhisperASRSegmentInferencer(
        model_name="openai/whisper-tiny",
        # model_name="bofenghuang/whisper-large-v2-cv11-german",
    )
    asr.build()
    with asr:
        out = asr.predict_transcribed_with_whisper_args(
            array, WhisperArgs(task="transcribe", language="en")
        )
        print(f"{out=}")

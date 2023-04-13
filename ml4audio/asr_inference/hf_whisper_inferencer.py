from dataclasses import dataclass, asdict, field

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
from misc_utils.dataclass_utils import UNDEFINED
from ml4audio.asr_inference.inference import (
    ASRAudioSegmentInferencer,
    StartEndTextsNonOverlap,
)
from ml4audio.asr_inference.whisper_inference import fix_start_end, WhisperArgs
from ml4audio.audio_utils.audio_io import ffmpeg_load_trim
from ml4audio.audio_utils.audio_segmentation_utils import (
    StartEnd,
    fix_segments_to_non_overlapping,
)


@dataclass
class HfPipelineWhisperASRSegmentInferencer(ASRAudioSegmentInferencer):

    model_name: str
    whisper_args: WhisperArgs = UNDEFINED
    chunk_length_s: float = 30.0  # see: https://huggingface.co/spaces/openai/whisper/discussions/67#63eb6ec2c6beb750e2cf47e9
    num_beams: int = 5
    hf_pipeline: AutomaticSpeechRecognitionPipeline = field(init=False, repr=False)

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
            model=self.model_name,
            chunk_length_s=self.chunk_length_s
            # , device=args.device
        )
        whisper_prompt_ids = whisper_pipeline.tokenizer.get_decoder_prompt_ids(
            language=self.whisper_args.language, task=self.whisper_args.task
        )
        whisper_pipeline.model.config.forced_decoder_ids = whisper_prompt_ids
        # print(f"{whisper_pipeline.model.config.max_length=}")
        # NB: decoding option
        # limit the maximum number of generated tokens to 225
        # whisper_pipeline.model.config.max_length = 225 + 1
        # sampling
        # pipe.model.config.do_sample = True
        # beam search
        whisper_pipeline.model.config.num_beams = self.num_beams
        # return
        whisper_pipeline.model.config.return_dict_in_generate = True  # TODO: why?
        # pipe.model.config.output_scores = True
        # pipe.model.config.num_return_sequences = 5

        self.hf_pipeline = whisper_pipeline  # noqa

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        del self.hf_pipeline

    @beartype
    def predict_transcribed_segments(
        self, audio_array: NumpyFloat1D
    ) -> StartEndTextsNonOverlap:
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
    """
    https://discuss.huggingface.co/t/support-for-asr-inference-on-longer-audiofiles-or-on-live-transcription/30464

    """
    file = "audiomonolith/tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.wav"
    array = ffmpeg_load_trim(file, sr=16000)

    asr = HfPipelineWhisperASRSegmentInferencer(
        model_name="openai/whisper-base",
        # model_name="bofenghuang/whisper-large-v2-cv11-german",
        whisper_args=WhisperArgs(task="transcribe", language="de"),
    )
    with asr:
        out = asr.predict_transcribed_segments(array)
        print(f"{out=}")

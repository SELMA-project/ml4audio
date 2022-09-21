from misc_utils.beartypes import NumpyFloat1DArray
from ml4audio.asr_inference.asr_chunk_infer_glue_pipeline import (
    Aschinglupi,
    calc_final_transcript,
    CompleteMessage,
)
from ml4audio.asr_inference.hfwav2vec2_asr_decode_inferencer import (
    HFASRDecodeInferencer,
)
from ml4audio.asr_inference.logits_inferencer.asr_logits_inferencer import (
    HfCheckpoint,
)
from ml4audio.asr_inference.logits_inferencer.hfwav2vec2_logits_inferencer import (
    HFWav2Vec2LogitsInferencer,
)
from ml4audio.asr_inference.transcript_gluer import (
    TranscriptGluer,
)
from ml4audio.audio_utils.aligned_transcript import AlignedTranscript
from ml4audio.audio_utils.audio_io import (
    convert_to_16bit_array,
    break_array_into_chunks,
)
from ml4audio.audio_utils.overlap_array_chunker import (
    OverlapArrayChunker,
    audio_messages_from_file,
    audio_messages_from_chunks,
)
from ml4audio.text_processing.ctc_decoding import GreedyDecoder


def wav2vec2_decode_inferencer(model="jonatasgrosman/wav2vec2-large-xlsr-53-german"):

    # TODO(tilo): WTF! I copypasted this from a test!
    # if not hasattr(request, "param"):
    expected_sample_rate = 16000
    # else:
    #     expected_sample_rate = request.param

    # model = "facebook/wav2vec2-base-960h"
    logits_inferencer = HFWav2Vec2LogitsInferencer(
        checkpoint=HfCheckpoint(
            name=model,
            model_name_or_path=model,
        ),
        input_sample_rate=expected_sample_rate,
    )
    asr = HFASRDecodeInferencer(
        logits_inferencer=logits_inferencer,
        decoder=GreedyDecoder(tokenizer_name_or_path=model),
    )
    asr.build()
    return asr


def aschinglupi_infer_file(
    wav_file: str, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-german"
) -> AlignedTranscript:
    """
    only for debugging -> loads asr-model with every call!
    """
    SR = 16000
    asr_input = list(audio_messages_from_file(wav_file, SR))
    return aschinglupi_infer_messages(asr_input, model_name, SR)


def aschinglupi_infer_array(
    array: NumpyFloat1DArray,
    model_name,
    target_sample_rate: int,
    chunk_duration: float = 0.1,
) -> AlignedTranscript:
    """
    only for debugging -> loads asr-model with every call!
    """
    a = convert_to_16bit_array(array)
    chunks_g = break_array_into_chunks(a, int(target_sample_rate * chunk_duration))
    messages_g = audio_messages_from_chunks("foobar", chunks_g)
    return aschinglupi_infer_messages(messages_g, model_name, target_sample_rate)


def aschinglupi_infer_messages(
    asr_input: CompleteMessage, model_name: str, SR
) -> AlignedTranscript:
    """
    only for debugging -> loads asr-model with every call!
    """
    window_dur = 8.0
    step_dur = 4.0
    streaming_asr = Aschinglupi(
        hf_asr_decoding_inferencer=wav2vec2_decode_inferencer(model_name),
        transcript_gluer=TranscriptGluer(),
        audio_bufferer=OverlapArrayChunker(
            chunk_size=int(window_dur * SR),
            minimum_chunk_size=int(1 * SR),  # one second!
            min_step_size=int(step_dur * SR),
        ),
    ).build()
    at = calc_final_transcript(streaming_asr, asr_input)
    return at

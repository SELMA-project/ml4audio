from beartype import beartype
from transformers import Wav2Vec2ForCTC
import torch
import argparse


@beartype
def convert_to_onnx(model_id_or_path: str, onnx_model_name):
    # based on : https://github.com/ccoreilly/wav2vec2-service/blob/master/convert_torch_to_onnx.py
    print(f"Converting {model_id_or_path} to onnx")
    # using: "torch_dtype=torch.float16" leads to "weight_norm_kernel" not implemented for 'Half'
    model = Wav2Vec2ForCTC.from_pretrained(model_id_or_path)
    audio_len = 250000

    x = torch.randn(1, audio_len, requires_grad=True)

    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_model_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {1: "audio_len"},  # variable length axes
            "output": {1: "audio_len"},
        },
    )


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    """
    TODO:
        use_external_data_format create extra file containing weights, this files absolute path on file system seems to be hard-coded in the onnx-file!
        so one cannot really copy it!
    """
    print("Starting quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        weight_type=QuantType.QInt8,  # better stay with default: QInt8
        use_external_data_format=True,  # to support big models (>2GB)
    )

    print(f"Quantized model saved to: {quantized_model_path}")


if __name__ == "__main__":
    """
    # seems to work!
    python ml4audio/asr_inference/pytorch_to_onnx_for_wav2vec.py --model jonatasgrosman/wav2vec2-large-xlsr-53-english
    
    # big model
    python ml4audio/asr_inference/pytorch_to_onnx_for_wav2vec.py --model jonatasgrosman/wav2vec2-xls-r-1b-english
    # leads to
    ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB: 3850629924

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ccoreilly/wav2vec2-large-100k-voxpopuli-catala",
        help="Model HuggingFace ID or path that will converted to ONNX",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to use also quantize the model or not",
    )
    args = parser.parse_args()

    model_id_or_path = args.model
    onnx_model_name = model_id_or_path.split("/")[-1] + ".onnx"
    convert_to_onnx(model_id_or_path, onnx_model_name)
    if args.quantize:
        quantized_model_name = model_id_or_path.split("/")[-1] + ".quant.onnx"
        quantize_onnx_model(onnx_model_name, quantized_model_name)

    import onnx

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)

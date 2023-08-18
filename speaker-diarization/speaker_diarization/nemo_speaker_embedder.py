import os.path
from dataclasses import field, dataclass
from typing import Any

import torch
from beartype import beartype
from tqdm import tqdm

from misc_utils.beartypes import (
    NeList,
    NumpyFloat1D,
)
from misc_utils.buildable import Buildable
from misc_utils.buildable_data import BuildableData, SlugStr
from misc_utils.dataclass_utils import UNDEFINED
from misc_utils.prefix_suffix import PrefixSuffix, BASE_PATHES
from misc_utils.processing_utils import iterable_to_batches
from ml4audio.audio_utils.nemo_utils import load_EncDecSpeakerLabelModel
from ml4audio.speaker_tasks.speaker_embedding_utils import SignalEmbedder
from nemo.collections.asr.models import EncDecSpeakerLabelModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@beartype
def embed_audio_chunks_with_nemo(
    speaker_model: EncDecSpeakerLabelModel,
    overlapping_chunks: NeList[NumpyFloat1D],
    batch_size: int,
) -> NeList[NumpyFloat1D]:
    """
    based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/examples/speaker_tasks/recognition/extract_speaker_embeddings.py

    based on: https://github.com/NVIDIA/NeMo/blob/aff169747378bcbcec3fc224748242b36205413f/nemo/collections/asr/models/clustering_diarizer.py#L329
    """
    if batch_size != 1:
        raise NotImplementedError("only batch size 1 is supported, don't ask me why!")
    speaker_model = speaker_model.to(DEVICE)
    speaker_model.eval()

    all_embs = []
    for test_batch in tqdm(
        iterable_to_batches(overlapping_chunks, batch_size=batch_size),
        desc="embedding with nemo",
    ):
        audio_tensors = [torch.from_numpy(x).to(DEVICE) for x in test_batch]
        audio_signal_len = torch.as_tensor([len(a) for a in audio_tensors]).to(DEVICE)
        no_need_for_padding_cause_all_have_same_len = (
            len(set([len(a) for a in test_batch])) == 1
        )
        assert no_need_for_padding_cause_all_have_same_len, set(
            [len(a) for a in test_batch]
        )
        audio_tensor = torch.concat([x.unsqueeze(0) for x in audio_tensors], dim=0)
        # probably based on: https://github.com/NVIDIA/NeMo/blob/4f06f3458b3d4d5e8ed3f5174d84e255a526321a/nemo/collections/asr/models/clustering_diarizer.py#L351
        with torch.no_grad():
            _, embs = speaker_model.forward(
                input_signal=audio_tensor, input_signal_length=audio_signal_len
            )
            emb_shape = embs.shape[-1]
            embs = embs.view(-1, emb_shape)
            all_embs.extend(embs.cpu().detach().numpy())

    return all_embs


@dataclass
class NemoAudioEmbedder(SignalEmbedder):
    model_name: str = UNDEFINED
    _speaker_model: EncDecSpeakerLabelModel = field(init=False, repr=False)

    base_dir: PrefixSuffix = field(
        default_factory=lambda: PrefixSuffix("cache_root", "MODELS/NEMO_MODELS")
    )

    @property
    def name(self) -> SlugStr:
        return f"{self.model_name}"

    @property
    def _is_data_valid(self) -> bool:
        return os.path.isfile(self.model_file)

    @property
    def model_file(self):
        return f"{self.data_dir}/model.nemo"

    def _build_data(self) -> Any:
        model = load_EncDecSpeakerLabelModel(self.model_name)
        model.save_to(self.model_file)

    def __enter__(self):
        self._speaker_model = EncDecSpeakerLabelModel.restore_from(
            restore_path=self.model_file
        )

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        del self._speaker_model

    @beartype
    def predict(self, arrays: NeList[NumpyFloat1D]) -> NeList[NumpyFloat1D]:
        return embed_audio_chunks_with_nemo(self._speaker_model, arrays, batch_size=1)


if __name__ == "__main__":
    BASE_PATHES["cache_root"] = "/tmp/cache_root"
    embedder = NemoAudioEmbedder(
        model_name="titanet_large",
    ).build()

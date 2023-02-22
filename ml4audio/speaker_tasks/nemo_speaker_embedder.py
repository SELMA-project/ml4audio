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
        iterable_to_batches(overlapping_chunks, batch_size=batch_size)
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
class NemoAudioEmbedder(SignalEmbedder, Buildable):
    model_name: str
    _speaker_model: EncDecSpeakerLabelModel = field(init=False, repr=False)

    def _build_self(self) -> Any:
        self._speaker_model = load_EncDecSpeakerLabelModel(self.model_name)

    @beartype
    def predict(self, arrays: NeList[NumpyFloat1D]) -> NeList[NumpyFloat1D]:
        return embed_audio_chunks_with_nemo(self._speaker_model, arrays, batch_size=1)

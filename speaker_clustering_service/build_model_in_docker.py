from pprint import pprint

from misc_utils.dataclass_utils import (
    to_dict,
)

from ml4audio.speaker_tasks.speaker_clusterer import UmascanSpeakerClusterer

if __name__ == "__main__":
    """
    maybe it acts as kind of sanity/integration test??
    it also downloads the model, some where to .cache folder
    """

    inferencer = UmascanSpeakerClusterer(model_name="ecapa_tdnn", metric="cosine").build()
    pprint(to_dict(inferencer))

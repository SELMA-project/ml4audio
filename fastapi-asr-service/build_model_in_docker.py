from pprint import pprint

from fastapi_asr_service.app.fastapi_asr_service_utils import load_asr_inferencer, \
    load_vad_inferencer
from misc_utils.dataclass_utils import (
    to_dict,
)

if __name__ == "__main__":
    """
    maybe it acts as kind of sanity/integration test??
    """

    asr_inferencer = load_asr_inferencer()
    vad = load_vad_inferencer()
    pprint(to_dict(asr_inferencer))
    pprint(to_dict(vad))

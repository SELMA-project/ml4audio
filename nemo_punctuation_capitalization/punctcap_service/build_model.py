from pathlib import Path

from nemo.collections.nlp.models import PunctuationCapitalizationModel

if __name__ == "__main__":
    model_files = [str(p) for p in Path("/code").rglob("model.nemo")]
    nemo_model = model_files[0]
    inferencer = PunctuationCapitalizationModel.restore_from(nemo_model)
    default_query = "deutsche welle sometimes abbreviated to dw is a german public state-owned international broadcaster funded by the german federal tax budget the service is available in 32 languages dws satellite"
    result = inferencer.add_punctuation_capitalization([default_query])
    print(f"{inferencer.cfg=}")
    print(f"{default_query=}")
    print(f"{result=}")

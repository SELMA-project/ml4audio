# language classification
* setup
```shell
pip install -r nemo_language_classification/requirements.txt
```
* train
```shell
export BASE_PATH=<YOUR_BASE_PATH>
export PYTHONPATH=${PWD}
export HF_DATASETS_CACHE=${BASE_PATH}/data/huggingface_cache/datasets
source secrets.env
python nemo_language_classification/finetune_lang_clf.py


```

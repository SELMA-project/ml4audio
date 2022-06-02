# punctuation via sequence tagging
### [NeMo tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Punctuation_and_Capitalization.ipynb)
* [NeMo punctuation_capitalization_train_evaluate](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py)
* condensed tutorial
```shell
# getting data
DATA_DIR={BASE_PATH}/data/cache/PROCESSED_DATA/TATOEBA/ProcessedPunctuationData-deu-25f198f90b3ae66e11e5c001e2ff24df9e77dba5/data
BASE_PATH=${PWD}

python $NEMO_ROOT/examples/nlp/token_classification/data/get_tatoeba_data.py --data_dir $DATA_DIR --num_sample 10000 --clean_dir
    
# model configuration
MODEL_CONFIG_YAML=$BASE_PATH/punctuation/conf/punctuation_capitalization_config.yaml
export PYTHONPATH=${PYTHONPATH}:{BASE_PATH}/iais_code/NeMo

python punctuation/punctuation_capitalization_train_evaluate.py \
    +do_testing=true \
    pretrained_model=punctuation_en_bert \
    model.train_ds.ds_item=$DATA_DIR \
    model.train_ds.tokens_in_batch=1500 \
    model.train_ds.text_file=$DATA_DIR/text_train.txt \
    model.train_ds.labels_file=$DATA_DIR/labels_train.txt \
    model.validation_ds.ds_item=$DATA_DIR \
    model.validation_ds.text_file=$DATA_DIR/text_dev.txt \
    model.validation_ds.labels_file=$DATA_DIR/labels_dev.txt \
    model.test_ds.ds_item=$DATA_DIR \
    model.test_ds.text_file=$DATA_DIR/text_dev.txt \
    model.test_ds.labels_file=$DATA_DIR/labels_dev.txt


Epoch 2:  99%|███████████▏ | 84/85 [00:07<00:00, 10.68it/s, loss=0.00611, v_num=1-17, lr=4.5e-7[NeMo I 2022-03-02 21:03:01 punctuation_capitalization_model:345] 
Punctuation report: ████████▎  | 2/3 [00:00<00:00,  5.45it/s]
    label                                                precision    recall       f1           support   
    O (label_id: 0)                                         99.68      99.78      99.73      13312
    , (label_id: 1)                                         89.67      86.77      88.20        310
    . (label_id: 2)                                         99.70      99.57      99.64       1646
    ? (label_id: 3)                                         99.21      98.95      99.08        380
    -------------------
    micro avg                                               99.48      99.48      99.48      15648
    macro avg                                               97.06      96.27      96.66      15648
    weighted avg                                            99.48      99.48      99.48      15648
    
[NeMo I 2022-03-02 21:03:01 punctuation_capitalization_model:346] Capitalization report: 
    label                                                precision    recall       f1           support   
    O (label_id: 0)                                         99.85      99.95      99.90      13264
    U (label_id: 1)                                         99.75      99.16      99.45       2384
    -------------------
    micro avg                                               99.83      99.83      99.83      15648
    macro avg                                               99.80      99.56      99.68      15648
    weighted avg                                            99.83      99.83      99.83      15648

```
### datasets
* [Helsinki-NLP/Tatoeba-Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/data/MonolingualData.md)
```shell
eng,deu,fra,por, spa
wget https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28/deu.tar

```
* wikibooks.txt.gz contains phrases like this: `Wassergekühlter 6-Zylinder turboaufgeladener ladeluftgekühlter Dieselmotor mit 8.277 ccm Hubraum und Bohrung x Hub 114 x 135,1 mm vom Typ C6T-830.`
    -> we might need proper written2spoken-text-formatting!
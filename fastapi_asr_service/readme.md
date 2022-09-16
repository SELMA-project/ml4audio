# fastapi ASR Service
### copy model-files into empty docker-image
```commandline
cd some-where-to-exported-model-folder
# bake into image
{ echo "FROM scratch" ; echo "COPY . ."; echo "CMD ['fake']"; } > Dockerfile && \
  export IMAGE=selmaproject/iais-asr-models:engdeu && \
  docker build -t $IMAGE . 
  
  # && docker image push $IMAGE
```
* example model-directory
```commandline
.
├── AM_MODELS
│   └── FinetunedCheckpoint-spanish-w2v-1b-970861b88cc53d764564a3757b7ef095923a9cd0
│       ├── dataclass.json
│       ├── Dockerfile
│       └── model
│           ├── config.json
│           ├── preprocessor_config.json
│           ├── pytorch_model.bin
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           └── vocab.json
├── ASR_INFERENCE
│   └── HfAsrPipelineFromLogitsInferencerDecoder-hfpipeline-spanish-w2v-1b-825b5066963f676d014dfcf92be9366c650afa46
│       └── dataclass.json
├── Dockerfile
└── LM_MODELS
    └── KenLMForPyCTCDecodeFromArpa-patrickvonplaten-spanish-ngram-lm-72c2589e580dfc61e1b419ba71b962b2488c1097
        ├── dataclass.json
        ├── lm.arpa.gz
        └── unigrams.txt.gz
```
* [what is this dataclass.json good for?](#dataclass-json)
### build docker-image
```commandline
LANG=deu
IMAGE=selmaproject/iais-asr-services:$LANG
DOCKER_BUILDKIT=1 docker build -f docker/fastapi_cpu/Dockerfile --build-arg MODEL_IMAGE=selmaproject/iais-asr-models:$LANG -t $IMAGE .
docker run -it --rm -v ${PWD}:/code --rm --shm-size 8G build_models:latest bash

curl -F ‘file=@path/to/local/file’ localhost:8000/transcribe
```
* image size: `6.7`GB; ~4GB: asr-model
```commandline
du -h / | grep -P "\dG|\d{3,5}M"
118M    /usr/lib
222M    /usr
114M    /venv/lib/python3.9/site-packages/scipy
1.5G    /venv/lib/python3.9/site-packages/torch/lib
1.6G    /venv/lib/python3.9/site-packages/torch
103M    /venv/lib/python3.9/site-packages/sklearn
2.3G    /venv/lib/python3.9/site-packages
2.3G    /venv/lib/python3.9
2.3G    /venv/lib
2.3G    /venv
3.6G    /model/AM_MODELS/FinetunedCheckpoint-spanish-w2v-1b-970861b88cc53d764564a3757b7ef095923a9cd0/model
3.6G    /model/AM_MODELS/FinetunedCheckpoint-spanish-w2v-1b-970861b88cc53d764564a3757b7ef095923a9cd0
3.6G    /model/AM_MODELS
257M    /model/LM_MODELS/KenLMForPyCTCDecodeFromArpa-patrickvonplaten-spanish-ngram-lm-72c2589e580dfc61e1b419ba71b962b2488c1097
257M    /model/LM_MODELS
3.9G    /model
6.4G    /

```
### run docker-image

```commandline
LANG_CODE=rus
docker run --rm -p 8000:8000 selmaproject/iais-asr-services:$LANG_CODE
```

# TODO
### async via ProcessPoolExecutor
* [see](https://testdriven.io/blog/fastapi-streamlit/)
```python
import asyncio

from concurrent.futures import ProcessPoolExecutor

from functools import partial

async def generate_remaining_models(models, image, name: str):
    executor = ProcessPoolExecutor()
    event_loop = asyncio.get_event_loop()
    await event_loop.run_in_executor(
        executor, partial(process_image, models, image, name)
    )


def process_image(models, image, name: str):
    for model in models:
        output, resized = inference.inference(models[model], image)
        name = name.split(".")[0]
        name = f"{name.split('_')[0]}_{models[model]}.jpg"
        cv2.imwrite(name, output)

@app.post("/{style}")
async def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    model = config.STYLES[style]
    start = time.time()
    output, resized = inference.inference(model, image)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    cv2.imwrite(name, output)
    models = config.STYLES.copy()
    del models[style]
    asyncio.create_task(generate_remaining_models(models, image, name))
    return {"name": name, "time": time.time() - start}
```
# dataclass json
* `what is this dataclass.json good for?`: mostly for documentation, here an example
* not really human-readable json that contains information about:
  * which python-classes where used: `ml4audio.audio_data.common_voice_datasets.CommonVoiceExtracted` for loading common-voice data
  * what data was used for model-finetuning:
    * `1603616` (1.6mio audios) -> 3600 hours seen by model during training
    * `SLR72` contains ~ 7.5 hours -> how much exactly was used during training is not (yet) "logged"->TODO! 
  * "learning_rate": 1e-05
  * "arpa_file" for LM -> "/some-where/foo/bar/data/LM_DATA/SPANISH_LM_DATA/hf_patrickvonplaten/kenLM.arpa"
  * "chunk_length_s" for chunked inference -> 16.0 seconds
```json
{
  "logits_inferencer": {
    "checkpoint": {
      "name": "spanish-w2v-1b",
      "model_name_or_path": "/behind/the/moon/data/cache/FINETUNE_TRAINING/HFWav2vec2FinetunerForDataproducer-spanish-w2v-1b-f9971cb4776d1b21e82f691b59cd510ed09b8916/output_dir",
      "finetune_master": {
        "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.hf_wav2vec2_finetuning.finetune_master_clients.FineTuneMaster",
        "name": "spanish-w2v-1b",
        "model_to_finetune": {
          "_python_dataclass_": "huggingface_wav2vec2_finetuning.base_model_for_finetuning.ModelIdentity",
          "name": "jonatas-w2v-1b-spanish",
          "model_name_or_path": "jonatasgrosman/wav2vec2-xls-r-1b-spanish"
        },
        "dataproducer_client": {
          "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.hf_wav2vec2_finetuning.finetune_master_clients.DataProducerClient",
          "task": {
            "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.hf_wav2vec2_finetuning.finetune_master_clients.EvalTrainDataProducer",
            "name": "eval_data",
            "edp": {
              "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.evaldata_producer.EvalDataProducer",
              "buffer_dir": {
                "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                "prefix_key": "train_worker_cache_root",
                "suffix": "spanish/EVAL_DATA",
                "prefix": "/behind/the/moon/data/cache"
              },
              "max_buffer_size": 2000,
              "bucket_size": 1,
              "dry_run": false,
              "cleanup_afterwards": false,
              "stats": null,
              "corpora": {
                "_python_dataclass_": "misc_utils.buildable.BuildableList",
                "data": [
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": {
                      "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                    },
                    "data": {
                      "_python_dataclass_": "ml4audio.audio_data.targz_asr_dataset.TarGzArrayText",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio.audio_data.mls_corpora.MLSIterableDataset",
                        "targztranscripts": {
                          "_python_dataclass_": "ml4audio.audio_data.mls_corpora.MLSTarGzTranscripts",
                          "base_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "cache_root",
                            "suffix": "RAW_DATA",
                            "prefix": "/some-where/foo/bar/data/cache"
                          },
                          "targz_file": "/some-where/data/corpora/Multilingual_LibriSpeech/mls_spanish.tar.gz"
                        },
                        "split": "dev"
                      },
                      "limit": null
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": {
                      "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                    },
                    "data": {
                      "_python_dataclass_": "ml4audio.audio_data.common_voice_datasets.CommonVoiceAuteda",
                      "sample_rate": 16000,
                      "raw_data": {
                        "_python_dataclass_": "ml4audio.audio_data.common_voice_datasets.CommonVoiceExtracted",
                        "base_dir": {
                          "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                          "prefix_key": "base_path",
                          "suffix": "data/ASR_DATA/COMMON_VOICE",
                          "prefix": "/some-where/foo/bar"
                        },
                        "targz_file": "/some-where/foo/bar/data/ASR_DATA/COMMON_VOICE/cv-corpus-10.0-2022-07-04-es.tar.gz"
                      },
                      "split_name": "test"
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  }
                ]
              },
              "perturbations": null,
              "max_audio_duration": 20.0,
              "num_workers": 0,
              "write_wav_file": false
            },
            "tdp": {
              "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.MaxLenDataProducer",
              "buffer_dir": {
                "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                "prefix_key": "train_worker_cache_root",
                "suffix": "spanish/spanish_k8s/TRAIN_DATA",
                "prefix": "/behind/the/moon/data/cache"
              },
              "max_buffer_size": 3200,
              "bucket_size": 32,
              "dry_run": false,
              "cleanup_afterwards": true,
              "stats": {
                "audio_lens_stats": {
                  "0": {
                    "count": 1603616.0,
                    "mean": 8.16175594001307,
                    "std": 4.682350575757143,
                    "min": 1.224,
                    "5%": 3.36,
                    "25%": 4.7786875,
                    "50%": 6.168,
                    "75%": 11.41,
                    "max": 20.0
                  }
                }
              },
              "corpora": {
                "_python_dataclass_": "misc_utils.buildable.BuildableList",
                "data": [
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 91.76841761111231,
                    "data": {
                      "_python_dataclass_": "ml4audio.audio_data.targz_asr_dataset.TarGzArrayText",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio.audio_data.mls_corpora.MLSIterableDataset",
                        "targztranscripts": {
                          "_python_dataclass_": "ml4audio.audio_data.mls_corpora.MLSTarGzTranscripts",
                          "base_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "cache_root",
                            "suffix": "RAW_DATA",
                            "prefix": "/some-where/foo/bar/data/cache"
                          },
                          "targz_file": "/some-where/data/corpora/Multilingual_LibriSpeech/mls_spanish.tar.gz"
                        },
                        "split": "train"
                      },
                      "limit": null
                    },
                    "factor": 0.1,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 10.003333333333398,
                    "data": {
                      "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.mediaspeech_corpora.MediaSpeechASRCorpus",
                        "stats": {
                          "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                        },
                        "lang": "es",
                        "name": "mediaspeech_es"
                      },
                      "min_duration": 0.5,
                      "max_duration": 20.0,
                      "min_text_len": 2,
                      "limit": null
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 64.58347421527628,
                    "data": {
                      "_python_dataclass_": "ml4audio.audio_data.common_voice_datasets.CommonVoiceAuteda",
                      "sample_rate": 16000,
                      "raw_data": {
                        "_python_dataclass_": "ml4audio.audio_data.common_voice_datasets.CommonVoiceExtracted",
                        "base_dir": {
                          "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                          "prefix_key": "base_path",
                          "suffix": "data/ASR_DATA/COMMON_VOICE",
                          "prefix": "/some-where/foo/bar"
                        },
                        "targz_file": "/some-where/foo/bar/data/ASR_DATA/COMMON_VOICE/cv-corpus-10.0-2022-07-04-es.tar.gz"
                      },
                      "split_name": "train"
                    },
                    "factor": 0.2,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 7.149439062500012,
                    "data": {
                      "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                        "stats": {
                          "count": 4374,
                          "duration_in_hours": 7.149416296296287,
                          "durations_percentiles": {
                            "count": 4374.0,
                            "mean": 5.88429324798049,
                            "std": 1.72429972957253,
                            "min": 1.6213333333333333,
                            "25%": 4.693333333333333,
                            "50%": 5.802666666666667,
                            "75%": 6.997333333333334,
                            "max": 12.544
                          }
                        },
                        "hf_dataset_name": "openslr",
                        "lang": "SLR71",
                        "split_name": "train",
                        "limit": null,
                        "hf_datasets_cache_dir": {
                          "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                          "prefix_key": "base_path",
                          "suffix": "huggingface_cache/datasets",
                          "prefix": "/some-where/foo/bar"
                        },
                        "name": "openslr-SLR71-train"
                      },
                      "min_duration": 0.5,
                      "max_duration": 20.0,
                      "min_text_len": 2,
                      "limit": null
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 7.5794974479167125,
                    "data": {
                      "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                        "stats": {
                          "count": 4903,
                          "duration_in_hours": 7.579472592592598,
                          "durations_percentiles": {
                            "count": 4903.0,
                            "mean": 5.565184852811203,
                            "std": 1.632367688882222,
                            "min": 1.9626666666666666,
                            "25%": 4.352,
                            "50%": 5.290666666666667,
                            "75%": 6.485333333333333,
                            "max": 13.824
                          }
                        },
                        "hf_dataset_name": "openslr",
                        "lang": "SLR72",
                        "split_name": "train",
                        "limit": null,
                        "hf_datasets_cache_dir": {
                          "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                          "prefix_key": "base_path",
                          "suffix": "huggingface_cache/datasets",
                          "prefix": "/some-where/foo/bar"
                        },
                        "name": "openslr-SLR72-train"
                      },
                      "min_duration": 0.5,
                      "max_duration": 20.0,
                      "min_text_len": 2,
                      "limit": null
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 9.219417638888903,
                    "data": {
                      "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                        "stats": {
                          "count": 5447,
                          "duration_in_hours": 9.219389629629623,
                          "durations_percentiles": {
                            "count": 5447.0,
                            "mean": 6.093226118352611,
                            "std": 1.6633928500338835,
                            "min": 2.304,
                            "25%": 4.949333333333334,
                            "50%": 5.888,
                            "75%": 7.082666666666666,
                            "max": 14.762666666666666
                          }
                        },
                        "hf_dataset_name": "openslr",
                        "lang": "SLR73",
                        "split_name": "train",
                        "limit": null,
                        "hf_datasets_cache_dir": {
                          "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                          "prefix_key": "base_path",
                          "suffix": "huggingface_cache/datasets",
                          "prefix": "/some-where/foo/bar"
                        },
                        "name": "openslr-SLR73-train"
                      },
                      "min_duration": 0.5,
                      "max_duration": 20.0,
                      "min_text_len": 2,
                      "limit": null
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 1.0027882465277778,
                    "data": {
                      "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                        "stats": {
                          "count": 617,
                          "duration_in_hours": 1.0027851851851846,
                          "durations_percentiles": {
                            "count": 617.0,
                            "mean": 5.8509346299297675,
                            "std": 1.899728185219763,
                            "min": 2.304,
                            "25%": 4.437333333333333,
                            "50%": 5.546666666666667,
                            "75%": 6.997333333333334,
                            "max": 12.8
                          }
                        },
                        "hf_dataset_name": "openslr",
                        "lang": "SLR74",
                        "split_name": "train",
                        "limit": null,
                        "hf_datasets_cache_dir": {
                          "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                          "prefix_key": "base_path",
                          "suffix": "huggingface_cache/datasets",
                          "prefix": "/some-where/foo/bar"
                        },
                        "name": "openslr-SLR74-train"
                      },
                      "min_duration": 0.5,
                      "max_duration": 20.0,
                      "min_text_len": 2,
                      "limit": null
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  },
                  {
                    "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                    "proportion": 4.814311111111106,
                    "data": {
                      "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                      "sample_rate": 16000,
                      "corpus": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                        "stats": {
                          "count": 3357,
                          "duration_in_hours": 4.814293645833357,
                          "durations_percentiles": {
                            "count": 3357.0,
                            "mean": 5.162781389633602,
                            "std": 1.5967645919012008,
                            "min": 1.9626666666666666,
                            "25%": 3.925333333333333,
                            "50%": 4.949333333333334,
                            "75%": 6.058666666666666,
                            "max": 14.250666666666667
                          }
                        },
                        "hf_dataset_name": "openslr",
                        "lang": "SLR75",
                        "split_name": "train",
                        "limit": null,
                        "hf_datasets_cache_dir": {
                          "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                          "prefix_key": "base_path",
                          "suffix": "huggingface_cache/datasets",
                          "prefix": "/some-where/foo/bar"
                        },
                        "name": "openslr-SLR75-train"
                      },
                      "min_duration": 0.5,
                      "max_duration": 20.0,
                      "min_text_len": 2,
                      "limit": null
                    },
                    "factor": 1.0,
                    "min_hours": 1
                  }
                ]
              },
              "perturbations": [
                {
                  "_python_dataclass_": "ml4audio.audio_data.nemo_perturbation.TranscodePerturbationDC",
                  "proba": 0.3
                },
                {
                  "_python_dataclass_": "ml4audio.audio_data.nemo_perturbation.ManyRandomSoxPerturbations",
                  "proba": 0.3
                }
              ],
              "max_audio_duration": 20.0,
              "num_workers": 4
            }
          },
          "teardown_sleep_time": 1.0,
          "rank": null,
          "queue_dir": {
            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
            "prefix_key": "train_worker_cache_root",
            "suffix": "spanish/JOB_QUEUE/DATA_PRODUCER",
            "prefix": "/behind/the/moon/data/cache"
          },
          "train_data_dir": {
            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
            "prefix_key": "train_worker_cache_root",
            "suffix": "spanish/spanish_k8s/TRAIN_DATA",
            "prefix": "/behind/the/moon/data/cache"
          }
        },
        "finetune_client": {
          "_python_dataclass_": "misc_utils.build_cache_elsewhere.FileLockQueuedCacheBuilder",
          "task": {
            "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.hf_wav2vec2_finetuning.hf_finetuner_with_indirect_dep.HFWav2vec2FinetunerForDataproducer",
            "BASE_PATH": "some-where",
            "model_args": {
              "_python_dataclass_": "huggingface_wav2vec2_finetuning.base_model_for_finetuning.ModelArgs",
              "model_to_finetune": {
                "_python_dataclass_": "huggingface_wav2vec2_finetuning.base_model_for_finetuning.ModelIdentity",
                "name": "jonatas-w2v-1b-spanish",
                "model_name_or_path": "jonatasgrosman/wav2vec2-xls-r-1b-spanish"
              },
              "tokenizer_name_or_path": null,
              "text_normalizer": "no_punct",
              "casing": {
                "_python_dataclass_": "ml4audio.text_processing.asr_text_normalization.Casing",
                "value": "1"
              },
              "new_vocab": null,
              "freeze_feature_encoder": true,
              "attention_dropout": 0.1,
              "activation_dropout": 0.1,
              "feat_proj_dropout": 0.1,
              "hidden_dropout": 0.1,
              "final_dropout": 0.0,
              "mask_time_prob": 0.05,
              "mask_time_length": 10,
              "mask_feature_prob": 0.0,
              "mask_feature_length": 10,
              "gradient_checkpointing": false,
              "layerdrop": 0.0,
              "ctc_loss_reduction": "mean",
              "do_normalize_audio": true
            },
            "data_args": {
              "_python_dataclass_": "huggingface_wav2vec2_finetuning.base_model_for_finetuning.DataArgs",
              "eval_metrics": [
                "wer"
              ],
              "use_auth_token": false,
              "unk_token": "<unk>",
              "pad_token": "<pad>",
              "word_delimiter_token": "|",
              "phoneme_language": null
            },
            "train_dataset": {
              "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_loading.file_consuming_dataset.FileConsumingDataset",
              "perturbations": null,
              "data_dir": {
                "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                "prefix_key": "train_worker_cache_root",
                "suffix": "spanish/spanish_k8s/TRAIN_DATA",
                "prefix": "/nfs-storage/data/cache"
              },
              "remove_after_read": true,
              "slow_down_at": 0.0
            },
            "eval_dataset": {
              "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_loading.file_reading_dataset.FileReadingDataset",
              "data_dir": {
                "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                "prefix_key": "train_worker_cache_root",
                "suffix": "spanish/EVAL_DATA",
                "prefix": "/nfs-storage/data/cache"
              }
            },
            "train_args": {
              "_python_dataclass_": "huggingface_wav2vec2_finetuning.huggingface_wav2vec2_finetuner.TrainArgs",
              "run_name": "spanish-w2v-1b",
              "overwrite_output_dir": true,
              "max_steps": 100000,
              "num_train_epochs": 1,
              "per_device_train_batch_size": 2,
              "per_device_eval_batch_size": 1,
              "learning_rate": 1e-05,
              "lr_scheduler_type": "linear",
              "warmup_steps": 2000,
              "evaluation_strategy": "steps",
              "save_steps": 20000,
              "eval_steps": 10000,
              "logging_steps": 1000,
              "save_total_limit": 3,
              "dataloader_num_workers": 8,
              "no_cuda": false,
              "fp16": true,
              "group_by_length": false,
              "ignore_data_skip": true,
              "do_train": true,
              "do_eval": true,
              "metric_for_best_model": "wer",
              "load_best_model_at_end": true,
              "greater_is_better": false,
              "early_stopping_patience": -1,
              "early_stopping_threshold": 0.001,
              "min_steps": 20000,
              "gradient_accumulation_steps": 8,
              "deepspeed": null
            },
            "overwrite_old_cache": true,
            "indirect_dependencies": {
              "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.hf_wav2vec2_finetuning.hf_finetuner_with_indirect_dep.FineTunersIndirectDependencies",
              "train_data_producer": {
                "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.MaxLenDataProducer",
                "buffer_dir": {
                  "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                  "prefix_key": "train_worker_cache_root",
                  "suffix": "spanish/spanish_k8s/TRAIN_DATA",
                  "prefix": "/behind/the/moon/data/cache"
                },
                "max_buffer_size": 3200,
                "bucket_size": 32,
                "dry_run": false,
                "cleanup_afterwards": true,
                "stats": null,
                "corpora": {
                  "_python_dataclass_": "misc_utils.buildable.BuildableList",
                  "data": [
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio.audio_data.targz_asr_dataset.TarGzArrayText",
                        "sample_rate": 16000,
                        "corpus": {
                          "_python_dataclass_": "ml4audio.audio_data.mls_corpora.MLSIterableDataset",
                          "targztranscripts": {
                            "_python_dataclass_": "ml4audio.audio_data.mls_corpora.MLSTarGzTranscripts",
                            "base_dir": {
                              "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                              "prefix_key": "cache_root",
                              "suffix": "RAW_DATA",
                              "prefix": "/some-where/foo/bar/data/cache"
                            },
                            "targz_file": "/some-where/data/corpora/Multilingual_LibriSpeech/mls_spanish.tar.gz"
                          },
                          "split": "train"
                        },
                        "limit": null
                      },
                      "factor": 0.1,
                      "min_hours": 1
                    },
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                        "sample_rate": 16000,
                        "corpus": {
                          "_python_dataclass_": "ml4audio_v2.speech_data.mediaspeech_corpora.MediaSpeechASRCorpus",
                          "stats": {
                            "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                          },
                          "lang": "es",
                          "name": "mediaspeech_es"
                        },
                        "min_duration": 0.5,
                        "max_duration": 20.0,
                        "min_text_len": 2,
                        "limit": null
                      },
                      "factor": 1.0,
                      "min_hours": 1
                    },
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio.audio_data.common_voice_datasets.CommonVoiceAuteda",
                        "sample_rate": 16000,
                        "raw_data": {
                          "_python_dataclass_": "ml4audio.audio_data.common_voice_datasets.CommonVoiceExtracted",
                          "base_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "base_path",
                            "suffix": "data/ASR_DATA/COMMON_VOICE"
                          },
                          "targz_file": "/some-where/foo/bar/data/ASR_DATA/COMMON_VOICE/cv-corpus-10.0-2022-07-04-es.tar.gz"
                        },
                        "split_name": "train"
                      },
                      "factor": 0.2,
                      "min_hours": 1
                    },
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                        "sample_rate": 16000,
                        "corpus": {
                          "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                          "stats": {
                            "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                          },
                          "hf_dataset_name": "openslr",
                          "lang": "SLR71",
                          "split_name": "train",
                          "limit": null,
                          "hf_datasets_cache_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "base_path",
                            "suffix": "huggingface_cache/transformers"
                          },
                          "name": "openslr-SLR71-train"
                        },
                        "min_duration": 0.5,
                        "max_duration": 20.0,
                        "min_text_len": 2,
                        "limit": null
                      },
                      "factor": 1.0,
                      "min_hours": 1
                    },
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                        "sample_rate": 16000,
                        "corpus": {
                          "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                          "stats": {
                            "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                          },
                          "hf_dataset_name": "openslr",
                          "lang": "SLR72",
                          "split_name": "train",
                          "limit": null,
                          "hf_datasets_cache_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "base_path",
                            "suffix": "huggingface_cache/transformers"
                          },
                          "name": "openslr-SLR72-train"
                        },
                        "min_duration": 0.5,
                        "max_duration": 20.0,
                        "min_text_len": 2,
                        "limit": null
                      },
                      "factor": 1.0,
                      "min_hours": 1
                    },
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                        "sample_rate": 16000,
                        "corpus": {
                          "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                          "stats": {
                            "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                          },
                          "hf_dataset_name": "openslr",
                          "lang": "SLR73",
                          "split_name": "train",
                          "limit": null,
                          "hf_datasets_cache_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "base_path",
                            "suffix": "huggingface_cache/transformers"
                          },
                          "name": "openslr-SLR73-train"
                        },
                        "min_duration": 0.5,
                        "max_duration": 20.0,
                        "min_text_len": 2,
                        "limit": null
                      },
                      "factor": 1.0,
                      "min_hours": 1
                    },
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                        "sample_rate": 16000,
                        "corpus": {
                          "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                          "stats": {
                            "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                          },
                          "hf_dataset_name": "openslr",
                          "lang": "SLR74",
                          "split_name": "train",
                          "limit": null,
                          "hf_datasets_cache_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "base_path",
                            "suffix": "huggingface_cache/transformers"
                          },
                          "name": "openslr-SLR74-train"
                        },
                        "min_duration": 0.5,
                        "max_duration": 20.0,
                        "min_text_len": 2,
                        "limit": null
                      },
                      "factor": 1.0,
                      "min_hours": 1
                    },
                    {
                      "_python_dataclass_": "ml4audio_v2.wav2vec2_finetuning.data_production.data_producer_vanilla.SizeAwareDataPart",
                      "proportion": {
                        "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                      },
                      "data": {
                        "_python_dataclass_": "ml4audio_v2.speech_data.asr_corpora.AscoAuteda",
                        "sample_rate": 16000,
                        "corpus": {
                          "_python_dataclass_": "ml4audio_v2.speech_data.huggingface_speech_corpora.HFSpeechASRCorpus",
                          "stats": {
                            "_python_dataclass_": "misc_utils.dataclass_utils._UNDEFINED"
                          },
                          "hf_dataset_name": "openslr",
                          "lang": "SLR75",
                          "split_name": "train",
                          "limit": null,
                          "hf_datasets_cache_dir": {
                            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
                            "prefix_key": "base_path",
                            "suffix": "huggingface_cache/transformers"
                          },
                          "name": "openslr-SLR75-train"
                        },
                        "min_duration": 0.5,
                        "max_duration": 20.0,
                        "min_text_len": 2,
                        "limit": null
                      },
                      "factor": 1.0,
                      "min_hours": 1
                    }
                  ]
                },
                "perturbations": [
                  {
                    "_python_dataclass_": "ml4audio.audio_data.nemo_perturbation.TranscodePerturbationDC",
                    "proba": 0.3
                  },
                  {
                    "_python_dataclass_": "ml4audio.audio_data.nemo_perturbation.ManyRandomSoxPerturbations",
                    "proba": 0.3
                  }
                ],
                "max_audio_duration": 20.0,
                "num_workers": 4
              }
            }
          },
          "teardown_sleep_time": 60.0,
          "rank": null,
          "queue_dir": {
            "_python_dataclass_": "misc_utils.prefix_suffix.PrefixSuffix",
            "prefix_key": "train_worker_cache_root",
            "suffix": "spanish/JOB_QUEUE/FINETUNER",
            "prefix": "/behind/the/moon/data/cache"
          }
        }
      }
    },
    "input_sample_rate": 16000,
    "resample_type": "kaiser_best",
    "do_normalize": true
  },
  "decoder": {
    "tokenizer_name_or_path": {
      "prefix_key": "am_models",
      "suffix": "FinetunedCheckpoint-spanish-w2v-1b-970861b88cc53d764564a3757b7ef095923a9cd0/model",
      "prefix": {
        "prefix_key": "cache_root",
        "suffix": "AM_MODELS",
        "prefix": "/some-where/foo/bar/data/cache"
      }
    },
    "lm_weight": 0.5,
    "beta": 1.5,
    "lm_data": {
      "transcript_normalizer": {
        "casing": {
          "value": "1"
        },
        "text_normalizer": "no_punct",
        "vocab": [
          "<pad>",
          "<s>",
          "</s>",
          "<unk>",
          "|",
          "'",
          "-",
          "a",
          "b",
          "c",
          "d",
          "e",
          "f",
          "g",
          "h",
          "i",
          "j",
          "k",
          "l",
          "m",
          "n",
          "o",
          "p",
          "q",
          "r",
          "s",
          "t",
          "u",
          "v",
          "w",
          "x",
          "y",
          "z",
          "á",
          "é",
          "í",
          "ñ",
          "ó",
          "ú",
          "ü"
        ]
      },
      "name": "patrickvonplaten-spanish-ngram-lm",
      "arpa_file": "/some-where/foo/bar/data/LM_DATA/SPANISH_LM_DATA/hf_patrickvonplaten/kenLM.arpa"
    },
    "num_best": 1,
    "beam_size": 100
  },
  "chunk_length_s": 16.0,
  "stride_length_s": null
}
```
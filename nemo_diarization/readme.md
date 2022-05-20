# Nvidia NeMo ClusteringDiarizer
![](images/speaker_visualization.png)
* [Speaker_Diarization_Inference.ipynb](https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb) 
```shell
python nemo_diarization_tutorial.py

labels=['0.298981 2.770311 A', '3.163901 5.147011 B'],reference=<pyannote.core.annotation.Annotation object at 0x7fa5fb3e8ca0>
name: ClusterDiarizer
num_workers: 4
sample_rate: 16000
batch_size: 64
diarizer:
  manifest_filepath: ???
  out_dir: ???
  oracle_vad: false
  collar: 0.25
  ignore_overlap: true
  vad:
    model_path: null
    external_vad_manifest: null
    parameters:
      window_length_in_sec: 0.15
      shift_length_in_sec: 0.01
      smoothing: median
      overlap: 0.875
      onset: 0.4
      offset: 0.7
      pad_onset: 0.05
      pad_offset: -0.1
      min_duration_on: 0.2
      min_duration_off: 0.2
      filter_speech_first: true
  speaker_embeddings:
    model_path: ???
    parameters:
      window_length_in_sec: 1.5
      shift_length_in_sec: 0.75
      multiscale_weights: null
      save_embeddings: false
  clustering:
    parameters:
      oracle_num_speakers: false
      max_num_speakers: 20
      enhanced_count_thres: 80
      max_rp_threshold: 0.25
      sparse_search_volume: 30

[NeMo I 2022-03-03 11:54:08 clustering_diarizer:130] Loading pretrained vad_marblenet model from NGC

...

postprocessing_params: {'window_length_in_sec': 0.15, 'shift_length_in_sec': 0.01, 'smoothing': 'median', 'overlap': 0.875, 'onset': 0.8, 'offset': 0.6, 'pad_onset': 0.05, 'pad_offset': -0.1, 'min_duration_on': 0.1, 'min_duration_off': 0.4, 'filter_speech_first': True}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.80it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.83it/s]
[NeMo W 2022-03-03 11:54:11 speaker_utils:335] cuda=False, using CPU for Eigen decompostion. This might slow down the clustering process.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 29.77it/s]

```

### outputs
* some rttm file!! -> whats that? never heard of this format!
```shell
SPEAKER an4_diarize_test 1   0.330   2.460 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER an4_diarize_test 1   3.210   1.890 <NA> <NA> speaker_0 <NA> <NA>
```
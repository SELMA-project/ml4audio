# Voice Activity Detection with Nvidie NeMo
* based on this [notebook](https://github.com/NVIDIA/NeMo/blob/v1.0.0/tutorials/asr/07_Online_Offline_Microphone_VAD_Demo.ipynb)
 
* nvividia says: `It is **not a recommended** way to do inference in production workflows. If you are interested in 
  production-level inference using NeMo ASR models, please sign-up to Jarvis early access program: https://developer.nvidia.com/nvidia-jarvis`

* setup
```shell
## Install dependencies
sudo apt-get install sox libsndfile1 ffmpeg portaudio19-dev
# ## Install NeMo
BRANCH = 'v1.0.0'
python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[asr]

## Install TorchAudio
pip install torchaudio>=0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

sudo apt-get install -y portaudio19-dev
pip install -r requirements.txt

```
![image](images/vad_demo.png)
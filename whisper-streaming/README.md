# live (streaming capable) ASR via Whisper
* https://github.com/collabora/WhisperLive
* https://github.com/davabase/whisper_real_time
* https://github.com/JonathanFly/faster-whisper-livestream-translator
* https://github.com/ufal/whisper_streaming
```commandline

conda activate py39_torch2
ENV_NAME=whisper_live
python -m venv ${ENVS_PATH}/${ENV_NAME} --system-site-packages
source ${ENVS_PATH}/$ENV_NAME/bin/activate

```
# see: https://github.com/docker/for-mac/issues/2155
# global arg ->  must declare it before every FROM or it will be local
ARG MODEL_IMAGE
FROM ${MODEL_IMAGE} as model_image

FROM python:3.9-bullseye AS dependencies
WORKDIR /code
ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        g++ \
        cmake \
        vim \
        # for testing \
        # libsndfile
        libsndfile1-dev \
        # portaudio
        portaudio19-dev python3-pyaudio \
        # ffmpeg
        ffmpeg libavcodec-extra \
       # sox \
        sox libsox-dev && \
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*


ENV PATH="/venv/bin:$PATH"
ENV PIP_INSTALL="/venv/bin/pip install --no-cache-dir --upgrade"

RUN apt-get update && apt-get install -y python3-venv
RUN python3 -m venv /venv && $PIP_INSTALL pip packaging setuptools
RUN $PIP_INSTALL torchaudio@https://download.pytorch.org/whl/cpu/torchaudio-0.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl
RUN $PIP_INSTALL install Cython

# to trigger re-run of following, "disable" caching, see: https://stackoverflow.com/questions/35134713/disable-cache-for-specific-run-commands
# use with: --build-arg CACHEBUST=$(date +%s)

ARG CACHEBUST=1
RUN echo "$CACHEBUST"

COPY requirements.txt requirements.txt
RUN $PIP_INSTALL -r requirements.txt

# ==================================================================
#                   BUILD MODELS - stage
# ------------------------------------------------------------------

FROM dependencies AS build_models
ENV CACHE_ROOT="/model"
COPY --from=model_image . /model
COPY build_model_in_docker.py /code/build_model_in_docker.py
COPY app/fastapi_asr_service_utils.py /code/fastapi_asr_service/app/fastapi_asr_service_utils.py
COPY app/vad_inference_postprocessing.yaml /code/app/vad_inference_postprocessing.yaml
COPY app/vad_multilingual_marblenet.nemo /code/app/vad_multilingual_marblenet.nemo
RUN python /code/build_model_in_docker.py
#
RUN rm -rf /venv/lib/python3.9/site-packages/sklearn/ensemble
RUN rm -rf /venv/lib/python3.9/site-packages/pynini.libs
RUN rm -rf /root/.cache/pip
RUN rm -rf /root/.cache/matplotlib

# ==================================================================
#                   PRODUCTION - stage
# ------------------------------------------------------------------
FROM python:3.9.13-slim-buster AS production
LABEL maintainer="Tilo Himmelsbach"
WORKDIR /code
ENV PATH="/venv/bin:$PATH"
ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        # libsndfile TODO: currently asr_logits_inferencer uses librosa to resample!!
        libsndfile1-dev \
        # portaudio
        portaudio19-dev python3-pyaudio \
        # ffmpeg
        ffmpeg libavcodec-extra \
        # sox \
        sox libsox-dev && \

    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

# maybe for better docker-caching copy from model-image here, this only works if build_models-stage does not modify the models!! does it? well it could!
#COPY --from=model_image . /model
COPY --from=build_models /model /model
COPY --from=build_models /venv /venv
COPY --from=build_models /root/.cache /root/.cache

ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# PYTHONFAULTHANDLER TODO: wasdatdenn?
ENV PYTHONFAULTHANDLER=1
ENV CACHE_ROOT="/model"
# ENV JINA_MP_START_METHOD=spawn

COPY app /code/app

CMD ["/bin/bash", "-c", "source /venv/bin/activate && \
           uvicorn app.main:app --host 0.0.0.0 --port 8000"]
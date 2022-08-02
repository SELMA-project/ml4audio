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
        vim && \
# ==================================================================
# clean up everything
# ------------------------------------------------------------------
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -U \
    pip packaging setuptools tqdm beartype Cython

COPY ./requirements.txt /code/requirements.txt
RUN /venv/bin/pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN rm -rf /venv/lib/python3.9/site-packages/sklearn/ensemble && \
    rm -rf /venv/lib/python3.9/site-packages/grpc

FROM dependencies AS build_models
ENV PATH="/venv/bin:$PATH"
COPY . /code/
RUN python /code/build_model.py
# TODO: cannot remove the following!
    #    rm -rf /venv/lib/python3.9/site-packages/tensorboard && \ -> pytorch-pytorch_lightning wants it!
    #     rm -rf /venv/lib/python3.9/site-packages/onnx -> nemo wants this! ->malparido!
    #     rm -rf /venv/lib/python3.9/site-packages/matplotlib -> nemo wants this! ->malparido!

FROM python:3.9-slim-bullseye AS production
LABEL maintainer="Tilo Himmelsbach"
ENV PATH="/venv/bin:$PATH"
WORKDIR /code
COPY --from=build_models /venv /venv
COPY --from=build_models /root/.cache /root/.cache

ENV PYTHONFAULTHANDLER=1
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000:8000
COPY . /code/

CMD ["/bin/bash", "-c", "source /venv/bin/activate && \
           uvicorn punctcap_fastapi_server:app --host 0.0.0.0 --port 8000"]


# fastapi ASR Service
### copy model-files into empty docker-image
```commandline
cd some-where-to-exported-model-folder
# bake into image
{ echo "FROM scratch" ; echo "COPY . ."; } > Dockerfile && \
  export IMAGE=selmaproject/iais-asr-models:spanish && \
  docker build -t $IMAGE . 
  
  # && docker image push $IMAGE
```
### build docker-image
```commandline
DOCKER_BUILDKIT=1 docker build -f docker/fastapi_cpu/Dockerfile --target build_models -t build_models .
IMAGE=selmaproject/iais-asr-services:spanish
DOCKER_BUILDKIT=1 docker build -f docker/fastapi_cpu/Dockerfile --build-arg MODEL_IMAGE=selmaproject/iais-asr-models:spanish -t $IMAGE .
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
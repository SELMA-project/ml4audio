# fastapi ASR Service

```commandline
cd languagemodel.german-default/model
# bake into image
{ echo "FROM scratch" ; echo "COPY . ."; } > Dockerfile && \
  export IMAGE=spanish_asr_model && \
  docker build -t $IMAGE . 
  
  # && docker image push $IMAGE
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
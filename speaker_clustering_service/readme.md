# speaker clustering service
### manually build docker-image
```commandline
IMAGE=selmaproject/iais-speaker-clustering-services:latest
DOCKER_BUILDKIT=1 docker build -f docker/fastapi_cpu/Dockerfile -t $IMAGE .
docker run -it --rm --shm-size 8G -p 8000:8000 $IMAGE bash

# for debugging
docker run -it -v ${PWD}:/code -v $CODE_DIR/misc-utils:/code/misc-utils -v $CODE_DIR/ml4audio:/code/ml4audio -v $CODE_DIR/misc-utils:/code/misc-utils -p 8001:8000 --rm $IMAGE bash

export PYTHONPATH=/code:/code/ml4audio && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

curl -F ‘file=@path/to/local/file’ localhost:8000/transcribe
# "production"
docker run -p 8001:8000 --rm $IMAGE
```

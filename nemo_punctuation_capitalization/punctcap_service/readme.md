# fastapi+NeMo-based punctuation&capitalization-service
```commandline
DOCKER_BUILDKIT=1 docker build -t punctcap .

docker run --rm -p 8000:8000 punctcap:latest

curl -H "Content-Type: application/json;charset=UTF-8" -X POST -d '{"text":"deutsche welle sometimes abbreviated to dw is a german public state-owned international broadcaster funded by the german federal tax budget the service is available in 32 languages dws satellite"}' http://localhost:8000/predict

text='Deutsche Welle, sometimes abbreviated to DW, is a German public, state-owned international broadcaster funded by the German federal tax budget. The service is available in 32 languages. DWs satellite'
pred="Deutsche Welle, sometimes abbreviated to Dw, is a German public state-owned international broadcaster funded by the German federal tax budget the service is available in 32 languages Dws satellite"

export MODEL=/nm-raid/audio/work/thimmelsba/data/cache/NEMO_PUNCTCAP_MODELS/NemoTrainedPunctuationCapitalizationModel-wiki-deu-dc4c2436c1d3f07b6a969ce1c28f43e19dc3221b/nemo_exp_dir/model.nemo

curl -F file=@"${MODEL}" localhost:8000/upload_modelfile
```

* debugging
```commandline
docker run -it --rm -p 8000:8000 -v ${PWD}:/code punctcap:latest bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
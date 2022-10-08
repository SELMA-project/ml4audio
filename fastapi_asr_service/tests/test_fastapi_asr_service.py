import os.path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app as webapp
from data_io.readwrite_files import read_file
from ml4audio.text_processing.asr_metrics import calc_cer


@pytest.fixture(scope="module")
def test_client() -> TestClient:
    with TestClient(webapp) as tc:
        yield tc


@pytest.fixture(scope="module")
def audio_file() -> str:
    return "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"


@pytest.fixture(scope="module")
def transcript_reference() -> str:
    return read_file(
        "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    )


def test_transcripe_endpoint(test_client, audio_file, transcript_reference):
    assert os.path.isdir(
        os.environ.get("CACHE_ROOT", "no-dir")
    ), "CACHE_ROOT where Aschinglupi got exported must be set as env-variable!"
    max_CER = 0.0

    resp = test_client.post(
        "/transcribe",
        json={
            "file": audio_file,
        },
    )

    assert resp.status_code == status.HTTP_200_OK
    hyp = resp.text

    cer = calc_cer([(hyp, transcript_reference)])
    assert cer <= max_CER
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "articles_sample.jsonl"


@pytest.fixture
def sample_path():
    assert SAMPLE.exists(), "python scripts/make_sample_data.py 를 먼저 실행하세요"
    return SAMPLE

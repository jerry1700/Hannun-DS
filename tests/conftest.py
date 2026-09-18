import json
from pathlib import Path

import pytest

from hannun.ingest.schema import expected_article_id

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "articles_sample.jsonl"


class FakeReducer:
    """umap 대신 쓰는 지도 — 학습 점은 앞 두 성분 그대로, transform 점은 100 을 더해 구별한다. pickle 되어야 한다."""

    def __init__(self, matrix):
        self.embedding_ = matrix[:, :2].astype("float32")

    def transform(self, matrix):
        return matrix[:, :2].astype("float32") + 100.0


@pytest.fixture
def sample_path():
    assert SAMPLE.exists(), "python scripts/make_sample_data.py 를 먼저 실행하세요"
    return SAMPLE


@pytest.fixture
def article_jsonl(tmp_path):
    """(별칭, 언론사, 발행 시각, 본문) 목록을 공통 기사 JSONL 로 써 경로를 돌려주는 팩토리.

    URL 은 별칭으로 만들고 article_id 는 실제 규칙으로 계산한다 — 테스트는 같은 식으로
    expected_article_id(언론사, "https://news.example.com/<별칭>") 로 id 를 되찾는다.
    """
    def write(articles, name="articles.jsonl"):
        lines = []
        for alias, publisher, published, content in articles:
            url = f"https://news.example.com/{alias}"
            lines.append(json.dumps({
                "schema_version": "1.0",
                "article_id": expected_article_id(publisher, url),
                "publisher_id": publisher,
                "publisher_name": publisher,
                "source_type": "RSS",
                "url": url,
                "title": f"{alias} 제목",
                "content": content,
                "author": None,
                "category": "정치",
                "category_str": None,
                "thumbnail_url": None,
                "language": "ko",
                "published_at": published,
            }, ensure_ascii=False))
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    return write

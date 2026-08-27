from datetime import timezone

import pytest
from pydantic import ValidationError

from hannun.ingest.schema import CommonArticle, SourceType, article_id_matches, expected_article_id


def base(**overrides) -> dict:
    d = {
        "schema_version": "1.0",
        "article_id": expected_article_id("yonhap", "https://www.yna.co.kr/view/1"),
        "publisher_id": "yonhap",
        "publisher_name": "연합뉴스",
        "source_type": "RSS",
        "url": "https://www.yna.co.kr/view/1",
        "title": "제목",
        "content": "본문",
        "author": "홍길동",
        "category": "POLITICS",
        "category_str": "정치>행정",
        "thumbnail_url": "https://x/y.jpg",
        "language": "ko",
        "published_at": "2026-08-20T05:30:00Z",
    }
    d.update(overrides)
    return d


def test_valid_record_parses():
    a = CommonArticle.model_validate(base())
    assert a.source_type is SourceType.RSS
    assert a.published_at.tzinfo == timezone.utc
    assert article_id_matches(a)


def test_offset_datetime_is_converted_to_utc():
    a = CommonArticle.model_validate(base(published_at="2026-08-21T02:00:00+09:00"))
    assert a.published_at.isoformat() == "2026-08-20T17:00:00+00:00"


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        CommonArticle.model_validate(base(published_at="2026-08-20T05:30:00"))


@pytest.mark.parametrize("field", ["content", "title", "url", "publisher_id"])
def test_empty_required_text_rejected(field):
    with pytest.raises(ValidationError) as ei:
        CommonArticle.model_validate(base(**{field: "   "}))
    assert field in str(ei.value)


def test_null_required_rejected():
    with pytest.raises(ValidationError):
        CommonArticle.model_validate(base(content=None))


def test_optional_empty_string_becomes_none():
    a = CommonArticle.model_validate(base(author="", category_str="  ", thumbnail_url=None))
    assert a.author is None and a.category_str is None and a.thumbnail_url is None


def test_unknown_source_type_rejected():
    with pytest.raises(ValidationError):
        CommonArticle.model_validate(base(source_type="API"))


def test_extra_fields_ignored():
    a = CommonArticle.model_validate(base(some_future_field=123))
    assert not hasattr(a, "some_future_field")


def test_content_is_not_modified():
    raw = "  앞뒤 공백과\n개행이 있는 본문  "
    a = CommonArticle.model_validate(base(content=raw))
    assert a.content == raw


def test_article_id_mismatch_detected():
    a = CommonArticle.model_validate(base(article_id="sha256:" + "0" * 64))
    assert not article_id_matches(a)


def test_article_id_prefix_optional():
    hex_only = expected_article_id("yonhap", "https://www.yna.co.kr/view/1").removeprefix("sha256:")
    a = CommonArticle.model_validate(base(article_id=hex_only.upper()))
    assert article_id_matches(a)

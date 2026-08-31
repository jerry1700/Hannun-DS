from datetime import timezone

import pytest
from pydantic import ValidationError

from hannun.ingest.schema import (
    CommonArticle, SourceType, article_id_matches, expected_article_id, normalize_url,
)


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


def test_html_source_type_accepted():
    a = CommonArticle.model_validate(base(source_type="HTML"))
    assert a.source_type is SourceType.HTML


def test_tracking_params_and_fragment_do_not_change_id():
    plain = "https://www.khan.co.kr/article/202608191645001/"
    tracked = plain + "?utm_source=khan_rss&utm_medium=rss&utm_campaign=politic_news#top"
    assert normalize_url(tracked) == plain
    assert expected_article_id("khan", tracked) == expected_article_id("khan", plain)


def test_identifying_query_is_kept_in_id():
    a = "http://www.dt.co.kr/contents.html?article_no=2024010102109919607007"
    b = "http://www.dt.co.kr/contents.html?article_no=2024010102109919607008"
    assert normalize_url(a) == a
    assert expected_article_id("dt", a) != expected_article_id("dt", b)


def test_article_id_matches_after_tracking_param_added():
    plain_id = expected_article_id("yonhap", "https://www.yna.co.kr/view/1")
    tracked = "https://www.yna.co.kr/view/1?utm_source=rss"
    a = CommonArticle.model_validate(base(url=tracked, article_id=plain_id))
    assert article_id_matches(a)


def base_v2(**overrides) -> dict:
    """개편된 article_v1(2026-08-31) 필드명 — DE 가 실제로 보내는 모양."""
    d = {
        "schema_version": "1.0",
        "article_id": expected_article_id("yonhap", "https://www.yna.co.kr/view/1"),
        "publisher_id": "yonhap",
        "company": "연합뉴스",
        "source_type": "RSS",
        "link": "https://www.yna.co.kr/view/1",
        "title": "제목",
        "article": "본문",
        "reporter": "홍길동",
        "category": "사회",
        "category_str": "뉴스 > 사회",
        "thumbnail_url": None,
        "language": "ko",
        "published": "2026-08-20T05:30:00Z",
        "crawled_at": "2026-08-20T05:31:00Z",
    }
    d.update(overrides)
    return d


def test_renamed_fields_map_to_internal_names():
    a = CommonArticle.model_validate(base_v2())
    assert a.publisher_name == "연합뉴스"
    assert a.url == "https://www.yna.co.kr/view/1"
    assert a.content == "본문"
    assert a.author == "홍길동"
    assert a.published_at.tzinfo == timezone.utc
    assert article_id_matches(a)


def test_renamed_payload_missing_body_rejected():
    with pytest.raises(ValidationError):
        CommonArticle.model_validate({k: v for k, v in base_v2().items() if k != "article"})


def test_new_tracking_params_do_not_change_id():
    plain = "https://n.news.example.com/article/001/0015000000"
    tracked = plain + "?ref=main&from=news&igshid=abc&page=2&date=20260820"
    assert normalize_url(tracked) == plain
    assert expected_article_id("ex", tracked) == expected_article_id("ex", plain)


def test_query_order_does_not_change_id():
    a = "https://news.sbs.co.kr/news/endPage.do?news_id=N9001&plink=COPYPASTE"
    b = "https://news.sbs.co.kr/news/endPage.do?plink=COPYPASTE&news_id=N9001"
    assert normalize_url(a) == normalize_url(b)
    assert expected_article_id("sbs", a) == expected_article_id("sbs", b)
